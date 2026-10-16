"""Causal behavioral features for channel and unified player histories.

Each output row describes information available at or before that event's
timestamp. Synthetic scenario labels are rejected, and demographic audit
attributes are returned by a separate function rather than model features.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


EXPECTED_CHANNELS = 3
GROUND_TRUTH_COLUMNS = frozenset({
    'scenario', 'injected_risk', 'network_group', 'scenario_type',
    'start_time', 'end_time', 'target_risk_category', 'ground_truth_label',
})
IDENTIFIER_COLUMNS = (
    'event_id', 'timestamp', 'channel', 'activity_entity_id',
    'canonical_player_id', 'proposed_canonical_player_id', 'source_player_id',
)
FEATURE_DOCUMENTATION = {
    'channel_*': 'Causal features using only the event source channel.',
    'unified_*': 'Causal features using all confirmed activity for the activity entity.',
    'observed_combined_activity_30d': 'Observed event count across channels in the trailing 30 days.',
    'cross_channel_deposit_trend': 'Seven-day deposit rate versus the preceding 23-day rate.',
    'channel_participation_change': 'Active channels in 7 days minus the preceding 23 days.',
    'channel_coverage_30d': 'Observed active channels divided by three; absence is not imputed as activity.',
    'peer_deposit_robust_z': 'Causal robust comparison with current behavioral peers.',
    'limited_history': 'True when fewer than 30 observed days or five observed records are available.',
}


def _safe_number(value):
    return float(value) if pd.notna(value) else np.nan


def _mad(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.nan, np.nan
    median = float(np.median(clean))
    return median, float(np.median(np.abs(clean - median)))


def _sum_if_observed(values: np.ndarray, observed: np.ndarray) -> float:
    known = values[observed & np.isfinite(values)]
    return float(known.sum()) if observed.any() else np.nan


def _session_key(row) -> str | None:
    if not bool(row.observed):
        return None
    if pd.notna(row.session_id):
        return str(row.session_id)
    return f'{row.channel}:{row.timestamp:%Y-%m-%d}'


def _causal_history_features(
    frame: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Calculate trailing and expanding features with timestamp-bounded masks."""
    records = {}
    loss_events = {'wager', 'payout', 'settlement', 'wager_observation', 'payout_observation'}

    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        group = group.sort_values(['timestamp', 'event_id'])
        timestamps = group.timestamp.to_numpy(dtype='datetime64[ns]')
        observed = group.observed.fillna(False).to_numpy(dtype=bool)
        deposits = pd.to_numeric(group.deposit_amount, errors='coerce').to_numpy(dtype=float)
        withdrawals = pd.to_numeric(group.withdrawal_amount, errors='coerce').to_numpy(dtype=float)
        wagers = pd.to_numeric(group.wager_amount, errors='coerce').to_numpy(dtype=float)
        losses = pd.to_numeric(group.net_loss, errors='coerce').to_numpy(dtype=float)
        relevant_loss = group.event_type.isin(loss_events).to_numpy(dtype=bool)
        sessions = np.array([_session_key(row) for row in group.itertuples()], dtype=object)
        devices = group.device_id.to_numpy(dtype=object)
        payments = group.payment_instrument_id.to_numpy(dtype=object)

        for position, (index, row) in enumerate(group.iterrows()):
            now = np.datetime64(row.timestamp.to_datetime64())
            eligible = timestamps <= now
            window_7 = eligible & (timestamps >= now - np.timedelta64(7, 'D'))
            window_30 = eligible & (timestamps >= now - np.timedelta64(30, 'D'))
            previous_23 = eligible & (timestamps < now - np.timedelta64(7, 'D')) \
                & (timestamps >= now - np.timedelta64(30, 'D'))
            history = eligible & observed
            observed_7 = window_7 & observed
            observed_30 = window_30 & observed
            observed_previous = previous_23 & observed

            deposit_events_7 = observed_7 & np.isfinite(deposits) & (deposits > 0)
            deposit_events_30 = observed_30 & np.isfinite(deposits) & (deposits > 0)
            withdrawal_events_7 = observed_7 & np.isfinite(withdrawals) & (withdrawals > 0)
            withdrawal_events_30 = observed_30 & np.isfinite(withdrawals) & (withdrawals > 0)

            historical_deposits = deposits[history & np.isfinite(deposits) & (deposits > 0)]
            median, mad = _mad(historical_deposits)
            zero_mad = bool(len(historical_deposits) >= 2 and pd.notna(mad) and mad < 1e-9)
            stability = np.nan
            if len(historical_deposits) >= 3 and pd.notna(median) and pd.notna(mad):
                stability = 1.0 / (1.0 + mad / max(abs(median), 1.0))

            deposits_7 = _sum_if_observed(deposits[window_7], observed[window_7])
            deposits_30 = _sum_if_observed(deposits[window_30], observed[window_30])
            deposits_previous = _sum_if_observed(deposits[previous_23], observed[previous_23])
            acceleration = np.nan
            if pd.notna(deposits_7) and pd.notna(deposits_previous) and deposits_previous > 1e-9:
                acceleration = (deposits_7 / 7) / (deposits_previous / 23) - 1

            sessions_7 = len(set(x for x in sessions[observed_7] if x is not None))
            sessions_30 = len(set(x for x in sessions[observed_30] if x is not None))
            sessions_previous = len(set(x for x in sessions[observed_previous] if x is not None))
            session_change = (
                (sessions_7 / 7) / (sessions_previous / 23) - 1
                if sessions_previous > 0 else np.nan
            )

            loss_total = int((window_30 & relevant_loss).sum())
            loss_known = int((window_30 & relevant_loss & observed & np.isfinite(losses)).sum())
            loss_reliable = loss_total > 0 and loss_known / loss_total >= .95
            net_loss_7 = losses[window_7 & relevant_loss & observed & np.isfinite(losses)].sum()
            net_loss_30 = losses[window_30 & relevant_loss & observed & np.isfinite(losses)].sum()
            if not loss_reliable:
                net_loss_7 = net_loss_30 = np.nan

            prior = timestamps < now
            prior_observed_times = timestamps[prior & observed]
            seconds_since_previous = (
                float((now - prior_observed_times.max()) / np.timedelta64(1, 's'))
                if len(prior_observed_times) else np.nan
            )
            prior_devices = set(str(x) for x in devices[prior] if pd.notna(x))
            prior_payments = set(str(x) for x in payments[prior] if pd.notna(x))
            current_device = str(row.device_id) if pd.notna(row.device_id) else None
            current_payment = str(row.payment_instrument_id) if pd.notna(row.payment_instrument_id) else None

            first_observed = timestamps[history].min() if history.any() else np.datetime64('NaT', 'ns')
            observation_days = (
                float((now - first_observed) / np.timedelta64(1, 'D'))
                if not np.isnat(first_observed) else np.nan
            )
            values = {
                f'{prefix}_deposit_frequency_7d': int(deposit_events_7.sum()),
                f'{prefix}_deposit_frequency_30d': int(deposit_events_30.sum()),
                f'{prefix}_deposit_amount_7d': deposits_7,
                f'{prefix}_deposit_amount_30d': deposits_30,
                f'{prefix}_withdrawal_frequency_7d': int(withdrawal_events_7.sum()),
                f'{prefix}_withdrawal_frequency_30d': int(withdrawal_events_30.sum()),
                f'{prefix}_wager_volume_7d': _sum_if_observed(wagers[window_7], observed[window_7]),
                f'{prefix}_wager_volume_30d': _sum_if_observed(wagers[window_30], observed[window_30]),
                f'{prefix}_net_loss_7d': net_loss_7,
                f'{prefix}_net_loss_30d': net_loss_30,
                f'{prefix}_net_loss_reliable_30d': loss_reliable,
                f'{prefix}_net_loss_observation_coverage_30d': (
                    loss_known / loss_total if loss_total else np.nan
                ),
                f'{prefix}_observed_events_7d': int(observed_7.sum()),
                f'{prefix}_observed_events_30d': int(observed_30.sum()),
                f'{prefix}_session_frequency_7d': sessions_7,
                f'{prefix}_session_frequency_30d': sessions_30,
                f'{prefix}_session_frequency_change': session_change,
                f'{prefix}_historical_median_spending': median,
                f'{prefix}_historical_spending_mad': mad,
                f'{prefix}_near_zero_mad': zero_mad,
                f'{prefix}_spending_acceleration': acceleration,
                f'{prefix}_behavioral_stability': stability,
                f'{prefix}_seconds_since_previous_event': seconds_since_previous,
                f'{prefix}_observation_days': observation_days,
                f'{prefix}_distinct_devices_30d': len(set(
                    str(x) for x in devices[window_30] if pd.notna(x)
                )),
                f'{prefix}_distinct_payment_instruments_30d': len(set(
                    str(x) for x in payments[window_30] if pd.notna(x)
                )),
                f'{prefix}_device_changed': (
                    current_device not in prior_devices if current_device is not None and prior_devices else False
                ),
                f'{prefix}_payment_instrument_changed': (
                    current_payment not in prior_payments if current_payment is not None and prior_payments else False
                ),
            }
            records[index] = values
    return pd.DataFrame.from_dict(records, orient='index').reindex(frame.index)


def _cross_channel_context(frame: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for _, group in frame.groupby('activity_entity_id', dropna=False, sort=False):
        group = group.sort_values(['timestamp', 'event_id'])
        timestamps = group.timestamp.to_numpy(dtype='datetime64[ns]')
        observed = group.observed.fillna(False).to_numpy(dtype=bool)
        channels = group.channel.to_numpy(dtype=object)
        amounts_missing = group[['amount', 'region']].isna().any(axis=1).to_numpy(dtype=bool)
        for index, row in group.iterrows():
            now = np.datetime64(row.timestamp.to_datetime64())
            eligible = timestamps <= now
            last_7 = eligible & (timestamps >= now - np.timedelta64(7, 'D'))
            last_30 = eligible & (timestamps >= now - np.timedelta64(30, 'D'))
            previous_23 = eligible & (timestamps < now - np.timedelta64(7, 'D')) \
                & (timestamps >= now - np.timedelta64(30, 'D'))
            observed_7 = last_7 & observed
            observed_30 = last_30 & observed
            observed_previous = previous_23 & observed
            active_7 = len(set(channels[observed_7]))
            active_30 = len(set(channels[observed_30]))
            active_previous = len(set(channels[observed_previous]))
            records_30 = int(last_30.sum())
            records[index] = {
                'active_channels_7d': active_7,
                'active_channels_30d': active_30,
                'active_channels_to_date': len(set(channels[eligible & observed])),
                'channel_participation_change': (
                    active_7 - active_previous if observed_previous.any() else np.nan
                ),
                'channel_coverage_30d': active_30 / EXPECTED_CHANNELS,
                'observed_record_fraction_30d': (
                    float(observed_30.sum() / records_30) if records_30 else np.nan
                ),
                'missing_field_fraction_30d': (
                    float(amounts_missing[last_30].sum() / records_30) if records_30 else np.nan
                ),
                'known_unobserved_records_30d': int((last_30 & ~observed).sum()),
            }
    return pd.DataFrame.from_dict(records, orient='index').reindex(frame.index)


def _peer_features(frame: pd.DataFrame, minimum_peers: int) -> pd.DataFrame:
    """Compare latest entity states using only peer information available by time T."""
    records = {}
    latest: dict[str, dict[str, float]] = defaultdict(dict)
    ordered = frame.sort_values(['timestamp', 'event_id'])
    for _, batch in ordered.groupby('timestamp', sort=True):
        for row in batch.itertuples():
            value = getattr(row, 'unified_deposit_amount_30d')
            if pd.notna(value):
                latest[row.behavioral_peer_group][row.activity_entity_id] = float(value)
        peer_stats = {}
        for peer_group in batch.behavioral_peer_group.unique():
            values = np.array(list(latest[peer_group].values()), dtype=float)
            median, mad = _mad(values)
            peer_stats[peer_group] = (values.size, median, mad)
        for index, row in batch.iterrows():
            count, median, mad = peer_stats[row.behavioral_peer_group]
            value = row.unified_deposit_amount_30d
            reliable = count >= minimum_peers and pd.notna(mad) and mad >= 1e-9
            records[index] = {
                'peer_entity_count': count,
                'peer_deposit_median_30d': median,
                'peer_deposit_mad_30d': mad,
                'peer_baseline_reliable': reliable,
                'peer_deposit_ratio': (
                    value / median
                    if pd.notna(value) and pd.notna(median) and abs(median) >= 1e-9
                    else np.nan
                ),
                'peer_deposit_robust_z': (
                    (value - median) / (1.4826 * mad)
                    if reliable and pd.notna(value) else np.nan
                ),
            }
    return pd.DataFrame.from_dict(records, orient='index').reindex(frame.index)


def fairness_audit_attributes(profiles: pd.DataFrame) -> pd.DataFrame:
    """Return protected/context attributes separately from operational features."""
    required = {'player_id', 'age_band', 'region'}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f'Missing fairness audit fields: {sorted(missing)}')
    return profiles[['player_id', 'age_band', 'region']].drop_duplicates('player_id').reset_index(drop=True)


def build_features(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    minimum_peer_entities: int = 5,
) -> pd.DataFrame:
    """Return a documented causal feature frame for training or inference.

    Rows are keyed by source event. Numeric missing values intentionally remain
    missing when history or observation quality is insufficient; companion
    coverage and reliability fields describe why.
    """
    leaked = GROUND_TRUTH_COLUMNS.intersection(events.columns) | GROUND_TRUTH_COLUMNS.intersection(
        profiles.columns
    )
    if leaked:
        raise ValueError(f'Ground-truth columns are not valid feature inputs: {sorted(leaked)}')
    required_events = {
        'event_id', 'timestamp', 'channel', 'event_type', 'activity_entity_id',
        'canonical_player_id', 'proposed_canonical_player_id', 'source_player_id',
        'identity_status', 'identity_confidence', 'deposit_amount', 'withdrawal_amount',
        'wager_amount', 'payout_amount', 'net_loss', 'amount', 'region', 'observed',
        'session_id', 'device_id', 'payment_instrument_id',
    }
    missing = required_events.difference(events.columns)
    if missing:
        raise ValueError(f'Missing normalized event fields: {sorted(missing)}')
    if minimum_peer_entities < 1:
        raise ValueError('minimum_peer_entities must be positive')
    required_profiles = {'player_id', 'account_open_date', 'gaming_preferences'}
    missing_profiles = required_profiles.difference(profiles.columns)
    if missing_profiles:
        raise ValueError(f'Missing profile fields: {sorted(missing_profiles)}')

    frame = events.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    frame = frame.sort_values(['timestamp', 'event_id']).reset_index(drop=True)
    channel_features = _causal_history_features(
        frame, ['activity_entity_id', 'channel'], 'channel'
    )
    unified_features = _causal_history_features(frame, ['activity_entity_id'], 'unified')
    cross_features = _cross_channel_context(frame)

    identifiers = frame[list(IDENTIFIER_COLUMNS) + ['identity_status', 'identity_confidence']].copy()
    features = pd.concat([identifiers, channel_features, unified_features, cross_features], axis=1)
    features['transaction_hour'] = frame.timestamp.dt.hour
    features['transaction_day_of_week'] = frame.timestamp.dt.dayofweek
    features['transaction_is_weekend'] = frame.timestamp.dt.dayofweek.ge(5)
    features['transaction_is_overnight'] = frame.timestamp.dt.hour.isin([0, 1, 2, 3, 4, 5])
    features['observed_combined_activity_30d'] = features.unified_observed_events_30d
    features['cross_channel_deposit_trend'] = features.unified_spending_acceleration
    features['channel_unified_deposit_difference_30d'] = (
        features.unified_deposit_amount_30d - features.channel_deposit_amount_30d
    )
    features['channel_share_of_unified_deposits_30d'] = (
        features.channel_deposit_amount_30d
        / features.unified_deposit_amount_30d.replace(0, np.nan)
    )

    profile_index = profiles.drop_duplicates('player_id').set_index('player_id')
    context_id = frame.canonical_player_id.fillna(frame.proposed_canonical_player_id).fillna(
        frame.source_player_id
    )
    open_dates = pd.to_datetime(context_id.map(profile_index.account_open_date))
    features['account_tenure_days'] = (frame.timestamp.dt.normalize() - open_dates).dt.days.astype(float)
    preferences = context_id.map(profile_index.gaming_preferences).fillna('unknown_behavior')
    features['behavioral_peer_group'] = preferences.astype(str)
    features['limited_history'] = (
        features.unified_observation_days.lt(30)
        | features.unified_observed_events_30d.lt(5)
        | features.unified_observation_days.isna()
    )
    peer = _peer_features(features, minimum_peer_entities)
    features = pd.concat([features, peer], axis=1)

    if {'age_band', 'region'}.intersection(features.columns):
        raise AssertionError('Fairness audit attributes entered operational features')
    features.attrs['feature_documentation'] = FEATURE_DOCUMENTATION
    features.attrs['non_model_identifier_columns'] = IDENTIFIER_COLUMNS
    return features
