"""Causal robust within-player behavioral change detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_behavioral_change(
    features: pd.DataFrame,
    signal_column: str = 'unified_deposit_amount_7d',
    minimum_history: int = 5,
    cusum_drift: float = .5,
) -> pd.DataFrame:
    """Score each event against only the player's strictly earlier signal history."""
    required = {'activity_entity_id', 'event_id', 'timestamp', signal_column}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'Missing behavioral-change fields: {sorted(missing)}')
    records = {}
    for _, group in features.groupby('activity_entity_id', dropna=False, sort=False):
        group = group.sort_values(['timestamp', 'event_id'])
        history = []
        cusum = 0.0
        for timestamp, batch in group.groupby('timestamp', sort=True):
            prior = np.asarray(history, dtype=float)
            prior = prior[np.isfinite(prior)]
            median = float(np.median(prior)) if len(prior) else np.nan
            mad = float(np.median(np.abs(prior - median))) if len(prior) else np.nan
            robust_scale = max(1.4826 * mad, abs(median) * .10, 1.0) if len(prior) else np.nan
            batch_values = pd.to_numeric(batch[signal_column], errors='coerce')
            for index, value in batch_values.items():
                sufficient = len(prior) >= minimum_history and pd.notna(value)
                robust_z = (float(value) - median) / robust_scale if sufficient else np.nan
                positive_change = max(robust_z, 0.0) if pd.notna(robust_z) else 0.0
                candidate_cusum = max(0.0, cusum + positive_change - cusum_drift)
                strength = max(positive_change, candidate_cusum / max(np.sqrt(len(prior)), 1))
                score = 100 * (1 - np.exp(-strength / 3.0)) if sufficient else np.nan
                records[index] = {
                    'behavioral_change_score': float(np.clip(score, 0, 100))
                    if pd.notna(score) else np.nan,
                    'behavioral_change_robust_z': robust_z,
                    'behavioral_change_baseline_median': median,
                    'behavioral_change_baseline_mad': mad,
                    'behavioral_change_history_count': len(prior),
                    'behavioral_change_sufficient': sufficient,
                }
            finite_batch = batch_values[np.isfinite(batch_values)].tolist()
            if finite_batch:
                # All same-timestamp rows use the same prior context, then update state.
                batch_change = max(
                    ((value - median) / robust_scale for value in finite_batch), default=0.0
                ) if len(prior) else 0.0
                cusum = max(0.0, cusum + max(batch_change, 0.0) - cusum_drift)
                history.extend(float(value) for value in finite_batch)
    return pd.DataFrame.from_dict(records, orient='index').reindex(features.index)


def score_latest_behavioral_change(
    features: pd.DataFrame,
    signal_column: str = 'unified_deposit_amount_7d',
    minimum_history: int = 5,
    cusum_drift: float = .5,
) -> pd.DataFrame:
    """Return the exact final change state per entity without materializing every row.

    This follows the same median/MAD/CUSUM update order as
    :func:`score_behavioral_change`. It is intended for dashboard snapshot
    histories where only current entity outputs are consumed.
    """
    required = {'activity_entity_id', 'event_id', 'timestamp', signal_column}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'Missing behavioral-change fields: {sorted(missing)}')
    records = {}
    ordered = features.sort_values(['activity_entity_id', 'timestamp', 'event_id'])
    for _, group in ordered.groupby('activity_entity_id', dropna=False, sort=False):
        history = []
        cusum = 0.0
        final_index = None
        final_record = None
        for index, value in pd.to_numeric(group[signal_column], errors='coerce').items():
            prior = np.asarray(history, dtype=float)
            prior = prior[np.isfinite(prior)]
            median = float(np.median(prior)) if len(prior) else np.nan
            mad = float(np.median(np.abs(prior - median))) if len(prior) else np.nan
            scale = max(1.4826 * mad, abs(median) * .10, 1.0) if len(prior) else np.nan
            sufficient = len(prior) >= minimum_history and pd.notna(value)
            robust_z = (float(value) - median) / scale if sufficient else np.nan
            positive = max(robust_z, 0.0) if pd.notna(robust_z) else 0.0
            candidate = max(0.0, cusum + positive - cusum_drift)
            strength = max(positive, candidate / max(np.sqrt(len(prior)), 1))
            score = 100 * (1 - np.exp(-strength / 3.0)) if sufficient else np.nan
            final_index = index
            final_record = {
                'behavioral_change_score': float(np.clip(score, 0, 100))
                if pd.notna(score) else np.nan,
                'behavioral_change_robust_z': robust_z,
                'behavioral_change_baseline_median': median,
                'behavioral_change_baseline_mad': mad,
                'behavioral_change_history_count': len(prior),
                'behavioral_change_sufficient': sufficient,
            }
            if pd.notna(value):
                batch_change = (float(value) - median) / scale if len(prior) else 0.0
                cusum = max(0.0, cusum + max(batch_change, 0.0) - cusum_drift)
                history.append(float(value))
        if final_index is not None:
            records[final_index] = final_record
    return pd.DataFrame.from_dict(records, orient='index')
