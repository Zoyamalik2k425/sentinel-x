"""Read-only dashboard projections and held-label evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from investigator_copilot import SIGNAL_LABELS
from scoring import prioritize_investigations
from signals import GROUND_TRUTH_COLUMNS


def friendly_entity_name(row: pd.Series) -> str:
    """Human-readable identity label without hiding linkage uncertainty."""
    if pd.notna(row.get('player_id')):
        return f"Player {row.get('player_id')}"
    entity = str(row.get('activity_entity_id', 'Unknown account'))
    parts = entity.split(':', 2)
    channel = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Source'
    account = parts[2] if len(parts) > 2 else entity
    status = str(row.get('identity_status', 'unlinked')).replace('_', ' ').title()
    return f'{status} · {channel} account {account}'


def ranked_queue(scored: pd.DataFrame, budget: int, case_records: dict | None = None) -> pd.DataFrame:
    """Create a display-ready ranked queue without mutating model outputs."""
    frame = scored.drop(columns=['priority_rank', 'selected_for_review'], errors='ignore')
    frame = prioritize_investigations(frame, budget)
    records = case_records or {}

    def reason(ledger):
        if not isinstance(ledger, dict):
            return 'No structured reason available'
        candidates = []
        for category in ('aml_fraud', 'safer_gambling'):
            for signal, detail in ledger.get(category, {}).items():
                if signal not in SIGNAL_LABELS or not isinstance(detail, dict):
                    continue
                contribution = detail.get('score_contribution')
                if contribution is not None and np.isfinite(contribution):
                    candidates.append((float(contribution), SIGNAL_LABELS[signal]))
        return max(candidates)[1] if candidates else 'No available contributing signal'

    def coverage(value):
        if not isinstance(value, dict):
            return np.nan
        return value.get('channel_coverage_30d')

    def quality(value):
        if not isinstance(value, dict):
            return 'unknown'
        return value.get('data_sufficiency_status', 'unknown')

    frame['primary_alert_reason'] = frame.signal_contributions.map(reason)
    frame['channel_coverage'] = frame.data_coverage.map(coverage)
    frame['data_quality'] = frame.data_coverage.map(quality)
    frame['investigation_status'] = frame.activity_entity_id.map(
        lambda entity: records.get(f'CASE-{entity}', {}).get('status', 'New')
    )
    frame['display_id'] = frame.apply(friendly_entity_name, axis=1)
    return frame


def _reviewed_players(queue: pd.DataFrame, budget: int) -> tuple[set[str], int]:
    reviewed = queue.sort_values('priority_rank').head(min(budget, len(queue)))
    players = set(reviewed.player_id.dropna().astype(str))
    return players, int(reviewed.player_id.isna().sum())


def evaluate_at_budget(
    queue: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    budget: int,
) -> dict[str, float | int]:
    """Compute fixed-budget metrics using held labels only after scoring."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(queue.columns)
    if leaked:
        raise ValueError(f'Ground-truth columns cannot be present in scored queue: {sorted(leaked)}')
    reviewed, unevaluable = _reviewed_players(queue, budget)
    positive = set(
        scenario_labels.loc[scenario_labels.ground_truth_label.astype(bool), 'player_id'].astype(str)
    )
    true_positives = reviewed & positive
    false_positives = reviewed - positive
    false_negatives = positive - reviewed
    return {
        'review_budget': min(budget, len(queue)),
        'evaluable_alerts': len(reviewed),
        'unlinked_alerts_not_labelled': unevaluable,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'precision_at_k': len(true_positives) / max(len(reviewed), 1),
        'recall_at_k': len(true_positives) / max(len(positive), 1),
    }


def performance_by_scenario(
    queue: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    reviewed, _ = _reviewed_players(queue, budget)
    labels = scenario_labels.copy()
    labels['reviewed'] = labels.player_id.astype(str).isin(reviewed)
    rows = []
    for scenario, group in labels.groupby('scenario_type', sort=True):
        positives = group.ground_truth_label.astype(bool)
        true_positives = int((group.reviewed & positives).sum())
        rows.append({
            'scenario_type': scenario,
            'players': len(group),
            'ground_truth_positives': int(positives.sum()),
            'reviewed': int(group.reviewed.sum()),
            'true_positives': true_positives,
            'false_positive_controls_reviewed': int((group.reviewed & ~positives).sum()),
            'recall_at_k': true_positives / max(int(positives.sum()), 1)
            if positives.any() else np.nan,
        })
    return pd.DataFrame(rows)


def performance_by_channel(
    queue: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    events: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    reviewed, _ = _reviewed_players(queue, budget)
    positive = set(
        scenario_labels.loc[scenario_labels.ground_truth_label.astype(bool), 'player_id'].astype(str)
    )
    rows = []
    observed = events[events.observed.fillna(False)] if 'observed' in events else events
    for channel, group in observed.groupby('channel', sort=True):
        column = 'canonical_player_id' if 'canonical_player_id' in group else 'player_id'
        players = set(group[column].dropna().astype(str))
        channel_positive = positive & players
        channel_reviewed = reviewed & players
        tp = channel_positive & channel_reviewed
        rows.append({
            'channel': channel,
            'observed_players': len(players),
            'positive_players': len(channel_positive),
            'reviewed_players': len(channel_reviewed),
            'precision_at_k': len(tp) / max(len(channel_reviewed), 1),
            'recall_at_k': len(tp) / max(len(channel_positive), 1),
            'false_positives': len(channel_reviewed - channel_positive),
            'false_negatives': len(channel_positive - channel_reviewed),
        })
    return pd.DataFrame(rows)


def fairness_audit(
    queue: pd.DataFrame,
    profiles: pd.DataFrame,
    field: str,
    budget: int,
) -> pd.DataFrame:
    if field not in {'age_band', 'region'}:
        raise ValueError('Fairness field must be age_band or region')
    reviewed, _ = _reviewed_players(queue, budget)
    frame = profiles[['player_id', field]].drop_duplicates('player_id').copy()
    frame['reviewed'] = frame.player_id.astype(str).isin(reviewed)
    audit = frame.groupby(field, dropna=False).agg(
        players=('player_id', 'count'),
        reviewed=('reviewed', 'sum'),
        review_rate=('reviewed', 'mean'),
    ).reset_index()
    audit['sample_status'] = np.where(audit.players.lt(30), 'low sample', 'sufficient sample')
    return audit

