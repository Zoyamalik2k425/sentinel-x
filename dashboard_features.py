"""Fast causal feature snapshots for the interactive dashboard.

The Phase 4 event-level builder remains the training/evaluation source of
truth. This module computes the same model-facing concepts at seven historical
cutoffs, using only events at or before each cutoff, so large local dashboard
runs do not rebuild a feature row for every transaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import GROUND_TRUTH_COLUMNS


EXPECTED_CHANNELS = 3
LOSS_EVENTS = {'wager', 'payout', 'settlement', 'wager_observation', 'payout_observation'}


def _group_series(frame, entities, column, aggregation='sum', fill=None):
    if frame.empty:
        result = pd.Series(index=entities, dtype=float)
    else:
        grouped = frame.groupby('activity_entity_id')[column]
        result = getattr(grouped, aggregation)().reindex(entities)
    return result.fillna(fill) if fill is not None else result


def _count_condition(frame, entities, condition):
    if frame.empty:
        return pd.Series(0, index=entities, dtype=float)
    return frame.loc[condition].groupby('activity_entity_id').size().reindex(entities, fill_value=0)


def _known_sum(frame, entities, column, observed_counts):
    values = pd.to_numeric(frame[column], errors='coerce')
    sums = frame.assign(_value=values).groupby('activity_entity_id')._value.sum(
        min_count=1
    ).reindex(entities)
    return sums.where(observed_counts.gt(0))


def _snapshot(events: pd.DataFrame, profiles: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(pd.Timestamp(cutoff).value, unit='ns')
    day_ns = 86_400_000_000_000
    cutoff_ns = cutoff.value
    event_ns = events.timestamp.astype('int64')
    eligible = events[event_ns.le(cutoff_ns)].copy()
    latest = eligible.sort_values(['timestamp', 'event_id']).groupby(
        'activity_entity_id', dropna=False
    ).tail(1).set_index('activity_entity_id')
    entities = latest.index
    eligible_ns = eligible.timestamp.astype('int64')
    start7_ns, start30_ns = cutoff_ns - 7 * day_ns, cutoff_ns - 30 * day_ns
    recent7 = eligible[eligible_ns.ge(start7_ns)]
    recent30 = eligible[eligible_ns.ge(start30_ns)]
    previous23 = recent30[recent30.timestamp.astype('int64').lt(start7_ns)]
    observed7 = recent7[recent7.observed.fillna(False)]
    observed30 = recent30[recent30.observed.fillna(False)]
    observed_previous = previous23[previous23.observed.fillna(False)]
    observed_history = eligible[eligible.observed.fillna(False)]

    frame = pd.DataFrame(index=entities)
    identity_columns = [
        'canonical_player_id', 'proposed_canonical_player_id', 'source_player_id',
        'identity_status', 'identity_confidence', 'channel',
    ]
    for column in identity_columns:
        frame[column] = latest[column]
    frame['event_id'] = [f'SNAPSHOT:{entity}:{cutoff.isoformat()}' for entity in entities]
    frame['timestamp'] = cutoff

    observed_count7 = observed7.groupby('activity_entity_id').size().reindex(entities, fill_value=0)
    observed_count30 = observed30.groupby('activity_entity_id').size().reindex(entities, fill_value=0)
    source_count30 = recent30.groupby('activity_entity_id').size().reindex(entities, fill_value=0)
    frame['unified_observed_events_7d'] = observed_count7
    frame['unified_observed_events_30d'] = observed_count30

    for days, observed_window, observed_counts in (
        (7, observed7, observed_count7), (30, observed30, observed_count30)
    ):
        deposits = pd.to_numeric(observed_window.deposit_amount, errors='coerce')
        withdrawals = pd.to_numeric(observed_window.withdrawal_amount, errors='coerce')
        frame[f'unified_deposit_frequency_{days}d'] = _count_condition(
            observed_window, entities, deposits.gt(0)
        )
        frame[f'unified_withdrawal_frequency_{days}d'] = _count_condition(
            observed_window, entities, withdrawals.gt(0)
        )
        frame[f'unified_deposit_amount_{days}d'] = _known_sum(
            observed_window, entities, 'deposit_amount', observed_counts
        )
        frame[f'unified_wager_volume_{days}d'] = _known_sum(
            observed_window, entities, 'wager_amount', observed_counts
        )

    deposits_previous = _known_sum(
        observed_previous,
        entities,
        'deposit_amount',
        observed_previous.groupby('activity_entity_id').size().reindex(entities, fill_value=0),
    )
    frame['unified_spending_acceleration'] = (
        (frame.unified_deposit_amount_7d / 7) / (deposits_previous / 23) - 1
    ).where(deposits_previous.gt(0))

    def sessions(window):
        if window.empty:
            return pd.Series(0, index=entities, dtype=float)
        keys = window.session_id.fillna(
            window.channel.astype(str) + ':' + window.timestamp.dt.strftime('%Y-%m-%d')
        )
        return window.assign(_session=keys).groupby('activity_entity_id')._session.nunique().reindex(
            entities, fill_value=0
        )

    sessions7, sessions30, sessions_previous = (
        sessions(observed7), sessions(observed30), sessions(observed_previous)
    )
    frame['unified_session_frequency_7d'] = sessions7
    frame['unified_session_frequency_30d'] = sessions30
    frame['unified_session_frequency_change'] = (
        (sessions7 / 7) / (sessions_previous / 23) - 1
    ).where(sessions_previous.gt(0))

    relevant = recent30.event_type.isin(LOSS_EVENTS)
    total_loss_records = recent30[relevant].groupby('activity_entity_id').size().reindex(
        entities, fill_value=0
    )
    known_loss = recent30[
        relevant & recent30.observed.fillna(False) & recent30.net_loss.notna()
    ]
    known_loss_count = known_loss.groupby('activity_entity_id').size().reindex(
        entities, fill_value=0
    )
    loss_coverage = known_loss_count / total_loss_records.replace(0, np.nan)
    reliable = total_loss_records.gt(0) & loss_coverage.ge(.95)
    loss30 = _group_series(known_loss, entities, 'net_loss', 'sum')
    known_loss7 = known_loss[known_loss.timestamp.astype('int64').ge(start7_ns)]
    loss7 = _group_series(known_loss7, entities, 'net_loss', 'sum')
    frame['unified_net_loss_7d'] = loss7.where(reliable)
    frame['unified_net_loss_30d'] = loss30.where(reliable)
    frame['unified_net_loss_reliable_30d'] = reliable
    frame['unified_net_loss_observation_coverage_30d'] = loss_coverage

    deposit_history = observed_history[
        pd.to_numeric(observed_history.deposit_amount, errors='coerce').gt(0)
    ]
    history_group = deposit_history.groupby('activity_entity_id').deposit_amount
    median = history_group.median().reindex(entities)
    history_with_median = deposit_history.assign(
        _median=deposit_history.activity_entity_id.map(median)
    )
    mad = history_with_median.assign(
        _deviation=(history_with_median.deposit_amount - history_with_median._median).abs()
    ).groupby('activity_entity_id')._deviation.median().reindex(entities)
    deposit_count = history_group.size().reindex(entities, fill_value=0)
    frame['unified_historical_median_spending'] = median
    frame['unified_historical_spending_mad'] = mad
    frame['unified_near_zero_mad'] = deposit_count.ge(2) & mad.lt(1e-9)
    frame['unified_behavioral_stability'] = (
        1 / (1 + mad / median.abs().clip(lower=1))
    ).where(deposit_count.ge(3))

    first_observed = observed_history.groupby('activity_entity_id').timestamp.min().reindex(entities)
    last_before = observed_history[observed_history.timestamp.lt(cutoff)].groupby(
        'activity_entity_id'
    ).timestamp.max().reindex(entities)
    frame['unified_observation_days'] = (cutoff - first_observed).dt.total_seconds() / 86_400
    frame['unified_seconds_since_previous_event'] = (
        cutoff - last_before
    ).dt.total_seconds()
    frame['unified_distinct_devices_30d'] = observed30.groupby(
        'activity_entity_id'
    ).device_id.nunique().reindex(entities, fill_value=0)
    frame['unified_distinct_payment_instruments_30d'] = observed30.groupby(
        'activity_entity_id'
    ).payment_instrument_id.nunique().reindex(entities, fill_value=0)

    active7 = observed7.groupby('activity_entity_id').channel.nunique().reindex(entities, fill_value=0)
    active30 = observed30.groupby('activity_entity_id').channel.nunique().reindex(entities, fill_value=0)
    active_previous = observed_previous.groupby('activity_entity_id').channel.nunique().reindex(
        entities, fill_value=0
    )
    frame['active_channels_7d'] = active7
    frame['active_channels_30d'] = active30
    frame['active_channels_to_date'] = observed_history.groupby(
        'activity_entity_id'
    ).channel.nunique().reindex(entities, fill_value=0)
    frame['channel_participation_change'] = (active7 - active_previous).where(
        observed_previous.groupby('activity_entity_id').size().reindex(entities, fill_value=0).gt(0)
    )
    frame['channel_coverage_30d'] = active30 / EXPECTED_CHANNELS
    frame['observed_record_fraction_30d'] = observed_count30 / source_count30.replace(0, np.nan)
    missing_fields = recent30[['amount', 'region']].isna().any(axis=1)
    frame['missing_field_fraction_30d'] = recent30.assign(
        _missing=missing_fields
    ).groupby('activity_entity_id')._missing.mean().reindex(entities)
    frame['known_unobserved_records_30d'] = recent30[
        ~recent30.observed.fillna(False)
    ].groupby('activity_entity_id').size().reindex(entities, fill_value=0)

    channel_deposits = observed30.groupby(
        ['activity_entity_id', 'channel']
    ).deposit_amount.sum().groupby('activity_entity_id').max().reindex(entities)
    frame['channel_deposit_amount_30d'] = channel_deposits
    frame['channel_unified_deposit_difference_30d'] = (
        frame.unified_deposit_amount_30d - channel_deposits
    )
    frame['channel_share_of_unified_deposits_30d'] = (
        channel_deposits / frame.unified_deposit_amount_30d.replace(0, np.nan)
    )
    frame['observed_combined_activity_30d'] = observed_count30
    frame['cross_channel_deposit_trend'] = frame.unified_spending_acceleration

    profile_index = profiles.drop_duplicates('player_id').set_index('player_id')
    context_id = frame.canonical_player_id.fillna(frame.proposed_canonical_player_id).fillna(
        frame.source_player_id
    )
    opened = pd.to_datetime(context_id.map(profile_index.account_open_date))
    frame['account_tenure_days'] = (cutoff.normalize() - opened).dt.days.astype(float)
    frame['behavioral_peer_group'] = context_id.map(
        profile_index.gaming_preferences
    ).fillna('unknown_behavior')

    peer_group = frame.groupby('behavioral_peer_group').unified_deposit_amount_30d
    peer_median = peer_group.transform('median')
    deviations = (frame.unified_deposit_amount_30d - peer_median).abs()
    peer_mad = deviations.groupby(frame.behavioral_peer_group).transform('median')
    peer_count = peer_group.transform('count')
    reliable_peer = peer_count.ge(5) & peer_mad.ge(1e-9)
    frame['peer_entity_count'] = peer_count
    frame['peer_deposit_median_30d'] = peer_median
    frame['peer_deposit_mad_30d'] = peer_mad
    frame['peer_baseline_reliable'] = reliable_peer
    frame['peer_deposit_ratio'] = (
        frame.unified_deposit_amount_30d / peer_median.replace(0, np.nan)
    )
    frame['peer_deposit_robust_z'] = (
        (frame.unified_deposit_amount_30d - peer_median) / (1.4826 * peer_mad)
    ).where(reliable_peer)
    frame['peer_deposit_percentile'] = frame.groupby(
        'behavioral_peer_group', dropna=False
    ).unified_deposit_amount_30d.rank(method='average', pct=True)
    frame['limited_history'] = frame.unified_observation_days.lt(30) | observed_count30.lt(5)
    frame['transaction_hour'] = cutoff.hour
    frame['transaction_day_of_week'] = cutoff.dayofweek
    frame['transaction_is_weekend'] = cutoff.dayofweek >= 5
    frame['transaction_is_overnight'] = cutoff.hour <= 5
    return frame.reset_index()


def build_dashboard_feature_history(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    snapshot_count: int = 7,
    snapshot_times: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return causal model-compatible snapshots for interactive inference."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(events.columns) | GROUND_TRUTH_COLUMNS.intersection(
        profiles.columns
    )
    if leaked:
        raise ValueError(f'Ground-truth columns are not valid feature inputs: {sorted(leaked)}')
    if snapshot_count < 6 and snapshot_times is None:
        raise ValueError('snapshot_count must be at least six for change detection')
    frame = events.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    start, end = frame.timestamp.min(), frame.timestamp.max()
    if snapshot_times is None:
        offsets = np.linspace(0, (end - start).total_seconds(), snapshot_count + 1)[1:]
        cutoffs = [start + pd.to_timedelta(offset, unit='s') for offset in offsets[:-1]] + [end]
    else:
        cutoffs = sorted(pd.Timestamp(value) for value in snapshot_times)
        if not cutoffs:
            raise ValueError('snapshot_times cannot be empty')
    snapshots = [_snapshot(frame, profiles, pd.Timestamp(cutoff)) for cutoff in cutoffs]
    return pd.concat(snapshots, ignore_index=True).sort_values(
        ['timestamp', 'event_id']
    ).reset_index(drop=True)

