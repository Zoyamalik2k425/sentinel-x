import numpy as np
import pandas as pd
import pytest

from ml.anomaly_detection import IsolationForestDetector
from ml.behavioral_change import score_behavioral_change, score_latest_behavioral_change
from ml.inference import ModelBundle, run_inference
from ml.peer_analysis import BehavioralPeerAnalyzer
from ml.train import chronological_split, fit_unsupervised_models, train_validate_test


def _feature_history(players=30, periods=12):
    rows = []
    start = pd.Timestamp('2026-01-01')
    for player in range(players):
        baseline = 40 + player * 2
        for period in range(periods):
            timestamp = start + pd.to_timedelta(period * 7, unit='D')
            escalation = 6.0 if player < 3 and period >= 9 else 1.0
            deposit_7 = baseline * escalation
            deposit_30 = deposit_7 * 3.2
            rows.append({
                'event_id': f'E-{player}-{period}',
                'timestamp': timestamp,
                'channel': 'online_casino',
                'activity_entity_id': f'PLAYER:P{player:04d}',
                'canonical_player_id': f'P{player:04d}',
                'proposed_canonical_player_id': None,
                'source_player_id': f'P{player:04d}',
                'identity_status': 'confirmed',
                'identity_confidence': 1.0,
                'unified_deposit_frequency_7d': 2 + period % 2,
                'unified_deposit_frequency_30d': 8 + period % 3,
                'unified_deposit_amount_7d': deposit_7,
                'unified_deposit_amount_30d': deposit_30,
                'unified_withdrawal_frequency_30d': period % 3,
                'unified_wager_volume_30d': deposit_30 * 1.4,
                'unified_net_loss_30d': np.nan if player == players - 1 else deposit_30 * .25,
                'unified_session_frequency_30d': 7 + period % 4,
                'unified_spending_acceleration': escalation - 1,
                'unified_behavioral_stability': .8,
                'unified_distinct_devices_30d': 1 + int(escalation > 1),
                'unified_distinct_payment_instruments_30d': 1,
                'active_channels_30d': 1 + (player % 3),
                'channel_participation_change': int(escalation > 1),
                'channel_coverage_30d': (1 + player % 3) / 3,
                'observed_record_fraction_30d': 1.0,
                'account_tenure_days': 180 + period * 7 + player,
                'unified_historical_median_spending': baseline,
                'unified_historical_spending_mad': max(baseline * .1, 1),
                'unified_observed_events_30d': 12 + period % 3,
                'unified_behavioral_stability': .8,
                'unified_observation_days': period * 7,
            })
    return pd.DataFrame(rows).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def _labels():
    return pd.DataFrame({
        'player_id': ['P0000', 'P0001', 'P0002'],
        'scenario_type': ['sudden_spending_escalation'] * 3,
        'start_time': [pd.Timestamp('2026-03-01')] * 3,
        'end_time': [pd.Timestamp('2026-12-31')] * 3,
        'target_risk_category': ['both'] * 3,
        'ground_truth_label': [True] * 3,
    })


def test_isolation_forest_handles_missing_values_and_returns_index():
    history = _feature_history()
    snapshots = history.groupby('activity_entity_id').head(1)
    detector = IsolationForestDetector(n_estimators=50).fit(snapshots)
    result = detector.predict(history.groupby('activity_entity_id').tail(1))

    assert result.anomaly_severity.between(0, 100).all()
    assert result.anomaly_feature_indicators.str.len().gt(0).all()


def test_behavioral_change_has_no_future_leakage():
    history = _feature_history(players=1, periods=8)
    before = score_behavioral_change(history)
    extended = history.copy()
    future = history.iloc[-1].copy()
    future['event_id'] = 'FUTURE'
    future['timestamp'] = pd.Timestamp('2027-01-01')
    future['unified_deposit_amount_7d'] = 1_000_000
    extended.loc[len(extended)] = future
    after = score_behavioral_change(extended).iloc[:len(history)]

    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_latest_behavioral_change_matches_full_history_final_rows():
    history = _feature_history(players=8, periods=10)
    full = score_behavioral_change(history)
    latest_indices = history.groupby('activity_entity_id').tail(1).index
    optimized = score_latest_behavioral_change(history)

    pd.testing.assert_frame_equal(
        full.loc[latest_indices].sort_index(), optimized.sort_index(), check_dtype=False
    )


def test_peer_analysis_uses_behavior_not_demographics():
    history = _feature_history()
    snapshots = history.groupby('activity_entity_id').head(1)
    analyzer = BehavioralPeerAnalyzer(n_clusters=4).fit(snapshots)
    result = analyzer.predict(history.groupby('activity_entity_id').tail(1))

    assert result.peer_deviation_score.between(0, 100).all()
    with pytest.raises(ValueError, match='Age and region'):
        BehavioralPeerAnalyzer(cluster_features=['age_band'])


def test_chronological_splits_do_not_overlap():
    splits = chronological_split(_feature_history())
    assert splits.train.timestamp.max() <= splits.train_end
    assert splits.validation.timestamp.min() > splits.train_end
    assert splits.validation.timestamp.max() <= splits.validation_end
    assert splits.test.timestamp.min() > splits.validation_end


def test_unsupervised_training_rejects_label_columns():
    history = _feature_history().assign(ground_truth_label=False)
    with pytest.raises(ValueError, match='cannot enter training'):
        fit_unsupervised_models(history)


def test_end_to_end_training_persistence_inference_and_evaluation(tmp_path):
    history = _feature_history()
    model_path = tmp_path / 'sentinel_x_models.joblib'
    bundle, metrics, coverage, outputs = train_validate_test(
        history, _labels(), model_path, random_state=11
    )
    restored = ModelBundle.load(model_path)
    repeated = run_inference(restored, history)

    assert model_path.exists()
    assert 0 <= bundle.alert_threshold <= 100
    assert {'precision', 'recall', 'f1', 'average_precision', 'roc_auc'}.issubset(metrics)
    assert set(coverage.coverage_level).issubset({'limited', 'partial', 'broad'})
    required = {
        'anomaly_severity', 'behavioral_change_score', 'peer_deviation_score',
        'top_contributing_feature_indicators', 'available_historical_context',
        'data_sufficiency_status',
    }
    assert required.issubset(outputs.columns)
    pd.testing.assert_series_equal(
        outputs.set_index('activity_entity_id').model_priority_index.sort_index(),
        repeated.set_index('activity_entity_id').model_priority_index.sort_index(),
    )
