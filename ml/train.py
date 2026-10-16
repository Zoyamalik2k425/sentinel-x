"""Chronological unsupervised training, validation thresholding, and evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .anomaly_detection import FORBIDDEN_FEATURES, IsolationForestDetector
from .inference import ModelBundle, run_inference
from .peer_analysis import BehavioralPeerAnalyzer


@dataclass(frozen=True)
class ChronologicalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def chronological_split(
    features: pd.DataFrame,
    train_fraction: float = .60,
    validation_fraction: float = .20,
) -> ChronologicalSplits:
    """Split strictly by event time; no random row mixing is permitted."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError('Chronological split fractions must leave non-empty test data')
    frame = features.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    times = np.sort(frame.timestamp.unique())
    if len(times) < 5:
        raise ValueError('At least five distinct timestamps are required')
    train_position = min(max(int(len(times) * train_fraction) - 1, 0), len(times) - 3)
    validation_position = min(
        max(int(len(times) * (train_fraction + validation_fraction)) - 1, train_position + 1),
        len(times) - 2,
    )
    train_end = pd.Timestamp(times[train_position])
    validation_end = pd.Timestamp(times[validation_position])
    train = frame[frame.timestamp <= train_end].copy()
    validation = frame[(frame.timestamp > train_end) & (frame.timestamp <= validation_end)].copy()
    test = frame[frame.timestamp > validation_end].copy()
    return ChronologicalSplits(train, validation, test, train_end, validation_end)


def _latest_entities(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(['timestamp', 'event_id']).groupby(
        'activity_entity_id', dropna=False
    ).tail(1)


def fit_unsupervised_models(
    training_features: pd.DataFrame,
    random_state: int = 42,
) -> tuple[IsolationForestDetector, BehavioralPeerAnalyzer]:
    """Fit without accepting a label table or label-bearing feature columns."""
    forbidden = FORBIDDEN_FEATURES.intersection(training_features.columns)
    if forbidden:
        raise ValueError(f'Ground-truth/demographic columns cannot enter training: {sorted(forbidden)}')
    snapshots = _latest_entities(training_features)
    anomaly = IsolationForestDetector(random_state=random_state).fit(snapshots)
    peers = BehavioralPeerAnalyzer(random_state=random_state).fit(snapshots)
    return anomaly, peers


def _truth_for_outputs(outputs: pd.DataFrame, scenario_labels: pd.DataFrame, as_of: pd.Timestamp):
    labels = scenario_labels.copy()
    labels['start_time'] = pd.to_datetime(labels.start_time)
    labels['end_time'] = pd.to_datetime(labels.end_time)
    positive = set(labels.loc[
        labels.ground_truth_label.astype(bool)
        & labels.start_time.le(as_of)
        & labels.end_time.ge(as_of),
        'player_id',
    ])
    return outputs.player_id.isin(positive).to_numpy(dtype=bool)


def select_validation_threshold(
    validation_outputs: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    as_of: pd.Timestamp,
) -> float:
    """Choose a validation F2 threshold, emphasizing recall over precision."""
    eligible = validation_outputs[validation_outputs.player_id.notna()].copy()
    truth = _truth_for_outputs(eligible, scenario_labels, as_of)
    scores = eligible.model_priority_index.to_numpy(dtype=float)
    if not truth.any():
        return float(np.quantile(scores, .92))
    candidates = np.unique(np.quantile(scores, np.linspace(.50, .99, 60)))
    best = (-1.0, -1.0, float(candidates[-1]))
    for threshold in candidates:
        predicted = scores >= threshold
        precision, recall, _, _ = precision_recall_fscore_support(
            truth, predicted, average='binary', zero_division=0
        )
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        candidate = (f2, recall, -float(threshold))
        if candidate > (best[0], best[1], -best[2]):
            best = (f2, recall, float(threshold))
    return best[2]


def evaluate_outputs(
    outputs: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    as_of: pd.Timestamp,
    threshold: float,
) -> dict[str, float | int]:
    eligible = outputs[outputs.player_id.notna()].copy()
    truth = _truth_for_outputs(eligible, scenario_labels, as_of)
    scores = eligible.model_priority_index.to_numpy(dtype=float)
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average='binary', zero_division=0
    )
    result = {
        'entities': len(eligible), 'positives': int(truth.sum()),
        'alerts': int(predicted.sum()), 'true_positives': int((truth & predicted).sum()),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'average_precision': float(average_precision_score(truth, scores)) if truth.any() else np.nan,
        'roc_auc': float(roc_auc_score(truth, scores)) if truth.any() and (~truth).any() else np.nan,
    }
    return result


def evaluate_by_coverage(
    outputs: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    as_of: pd.Timestamp,
    threshold: float,
) -> pd.DataFrame:
    frame = outputs.copy()
    frame['coverage_level'] = pd.cut(
        frame.channel_coverage_30d,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=['limited', 'partial', 'broad'],
    )
    rows = []
    for level, group in frame.groupby('coverage_level', observed=True):
        metrics = evaluate_outputs(group, scenario_labels, as_of, threshold)
        metrics['coverage_level'] = str(level)
        rows.append(metrics)
    return pd.DataFrame(rows)


def train_validate_test(
    features: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    model_path: str | Path,
    random_state: int = 42,
) -> tuple[ModelBundle, dict, pd.DataFrame, pd.DataFrame]:
    """Fit historical data, tune on validation, evaluate once on held-out test."""
    splits = chronological_split(features)
    anomaly, peers = fit_unsupervised_models(splits.train, random_state)
    provisional = ModelBundle(anomaly, peers, 100.0, splits.train_end, splits.validation_end)
    history_through_validation = features[features.timestamp <= splits.validation_end]
    validation_outputs = run_inference(provisional, history_through_validation, splits.validation_end)
    threshold = select_validation_threshold(validation_outputs, scenario_labels, splits.validation_end)
    bundle = ModelBundle(anomaly, peers, threshold, splits.train_end, splits.validation_end)
    bundle.save(model_path)

    test_end = pd.to_datetime(features.timestamp).max()
    test_outputs = run_inference(bundle, features, test_end)
    test_metrics = evaluate_outputs(test_outputs, scenario_labels, test_end, threshold)
    coverage_metrics = evaluate_by_coverage(test_outputs, scenario_labels, test_end, threshold)
    return bundle, test_metrics, coverage_metrics, test_outputs


def _main():
    parser = argparse.ArgumentParser(description='Train and evaluate local Sentinel X anomaly models.')
    parser.add_argument('--players', type=int, default=200)
    parser.add_argument('--events', type=int, default=4_000)
    parser.add_argument('--days', type=int, default=182)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', default='artifacts/phase5_models.joblib')
    args = parser.parse_args()

    from cross_channel import integrate_channels
    from data_gen import generate_dataset
    from feature_engineering import build_features

    data = generate_dataset(args.players, args.days, args.events, args.seed)
    integrated = integrate_channels(
        data.players, data.online_casino, data.sports_betting, data.land_based,
        data.start_time, data.end_time,
    )
    features = build_features(integrated.events, data.players)
    bundle, metrics, coverage, outputs = train_validate_test(
        features, data.scenario_labels, args.model_path, args.seed
    )
    print(f'Model artifact: {Path(args.model_path).resolve()}')
    print(f'Validation-selected threshold: {bundle.alert_threshold:.3f}')
    print('Held-out test metrics:')
    print(json.dumps(metrics, indent=2))
    print('Coverage comparison:')
    print(coverage.to_string(index=False))
    print('Highest current model outputs:')
    print(outputs.sort_values('model_priority_index', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    _main()
