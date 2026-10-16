"""Cross-channel normalization and conservative synthetic identity resolution.

The schemas and matching evidence are fictional project constructs. They do
not represent OLG systems. Probable or conflicting matches are never merged
into a confirmed canonical player history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CHANNELS = ('online_casino', 'sportsbook', 'land_casino')
MATCH_STATUSES = ('confirmed', 'probable', 'unlinked', 'conflicting')
COMMON_EVENT_COLUMNS = (
    'event_id', 'source_event_id', 'channel', 'timestamp', 'event_type',
    'source_account_id', 'source_player_id', 'canonical_player_id',
    'proposed_canonical_player_id', 'activity_entity_id', 'identity_status',
    'identity_confidence', 'deposit_amount', 'withdrawal_amount',
    'wager_amount', 'payout_amount', 'net_loss', 'amount', 'region',
    'observed', 'transaction_reference',
)


@dataclass(frozen=True)
class CrossChannelResult:
    identity_map: pd.DataFrame
    events: pd.DataFrame
    channel_summaries: pd.DataFrame
    unified_summaries: pd.DataFrame
    completeness: pd.DataFrame
    indicators: pd.DataFrame
    duplicate_records_removed: int


def build_identity_evidence(
    online_casino: pd.DataFrame,
    sports_betting: pd.DataFrame,
    land_based: pd.DataFrame,
) -> pd.DataFrame:
    """Build fictional evidence records from each source's available identifiers."""
    online = online_casino[['account_id', 'player_id']].drop_duplicates().rename(
        columns={'account_id': 'source_account_id', 'player_id': 'asserted_player_id'}
    )
    online = online.assign(
        channel='online_casino', synthetic_match_confidence=1.0,
        evidence='authenticated online-casino account', conflicting_identifier=False,
    )
    sports = sports_betting[['sports_account_id', 'player_id']].drop_duplicates().rename(
        columns={'sports_account_id': 'source_account_id', 'player_id': 'asserted_player_id'}
    )
    sports = sports.assign(
        channel='sportsbook', synthetic_match_confidence=1.0,
        evidence='authenticated sportsbook account', conflicting_identifier=False,
    )

    land = land_based.copy()
    land['source_account_id'] = np.where(
        land.loyalty_id.notna(),
        land.loyalty_id,
        np.where(
            land.patron_id.notna(),
            'PATRON-' + land.patron_id.astype(str),
            'UNLINKED-' + land.event_id.astype(str),
        ),
    )
    land = land.rename(columns={'patron_id': 'asserted_player_id'})
    land['synthetic_match_confidence'] = land.linkage_confidence.where(
        land.asserted_player_id.notna(), 0.0
    )
    land['evidence'] = np.where(
        land.loyalty_id.notna(), 'optional loyalty identifier',
        np.where(land.asserted_player_id.notna(), 'patron-linkage evidence', 'no player linkage'),
    )
    land['conflicting_identifier'] = False
    land = land[[
        'channel', 'source_account_id', 'asserted_player_id',
        'synthetic_match_confidence', 'evidence', 'conflicting_identifier',
    ]]
    return pd.concat([online, sports, land], ignore_index=True)


def resolve_identities(
    evidence: pd.DataFrame,
    canonical_player_ids: Iterable[str],
    confirmed_threshold: float = .95,
    probable_threshold: float = .60,
) -> pd.DataFrame:
    """Resolve source accounts without merging uncertain matches.

    ``canonical_player_id`` is populated only for confirmed matches. Probable
    matches retain a ``proposed_canonical_player_id`` for investigator review.
    """
    required = {'channel', 'source_account_id', 'asserted_player_id', 'synthetic_match_confidence'}
    missing = required.difference(evidence.columns)
    if missing:
        raise ValueError(f'Missing identity evidence fields: {sorted(missing)}')
    canonical = set(canonical_player_ids)
    working = evidence.copy()
    if 'conflicting_identifier' not in working:
        working['conflicting_identifier'] = False
    if 'evidence' not in working:
        working['evidence'] = 'synthetic matching evidence'

    rows = []
    for (channel, account), group in working.groupby(
        ['channel', 'source_account_id'], dropna=False, sort=True
    ):
        asserted = sorted(set(group.asserted_player_id.dropna().astype(str)))
        explicit_conflict = bool(group.conflicting_identifier.fillna(False).any())
        conflict = explicit_conflict or len(asserted) > 1
        confidence = float(np.clip(group.synthetic_match_confidence.fillna(0).mean(), 0, 1))
        proposed = asserted[0] if len(asserted) == 1 and asserted[0] in canonical else None
        if conflict:
            status, resolved = 'conflicting', None
        elif proposed is None:
            status, resolved = 'unlinked', None
        elif confidence >= confirmed_threshold:
            status, resolved = 'confirmed', proposed
        elif confidence >= probable_threshold:
            status, resolved = 'probable', None
        else:
            status, resolved, proposed = 'unlinked', None, None
        rows.append({
            'channel': channel,
            'source_account_id': account,
            'canonical_player_id': resolved,
            'proposed_canonical_player_id': proposed,
            'identity_status': status,
            'identity_confidence': confidence,
            'identity_evidence': ' | '.join(sorted(set(group.evidence.dropna().astype(str)))),
            'asserted_player_ids': '|'.join(asserted),
        })
    return pd.DataFrame(rows)


def _source_accounts(land: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            land.loyalty_id.notna(), land.loyalty_id,
            np.where(
                land.patron_id.notna(), 'PATRON-' + land.patron_id.astype(str),
                'UNLINKED-' + land.event_id.astype(str),
            ),
        ),
        index=land.index,
    )


def _attach_identity(frame: pd.DataFrame, identity_map: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(
        identity_map,
        on=['channel', 'source_account_id'],
        how='left',
        validate='many_to_one',
    )
    if merged.identity_status.isna().any():
        raise ValueError('Every source account must have an identity-map record')
    confirmed = merged.identity_status.eq('confirmed')
    merged['activity_entity_id'] = np.where(
        confirmed,
        'PLAYER:' + merged.canonical_player_id.fillna('').astype(str),
        'SOURCE:' + merged.channel.astype(str) + ':' + merged.source_account_id.astype(str),
    )
    return merged


def normalize_channel_events(
    online_casino: pd.DataFrame,
    sports_betting: pd.DataFrame,
    land_based: pd.DataFrame,
    identity_map: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize distinct sources while retaining channel and source identifiers."""
    online = online_casino.copy()
    online['source_account_id'] = online.account_id
    online['source_player_id'] = online.player_id
    online['source_event_id'] = online.event_id
    online['payout_amount_common'] = online.payout_amount
    online['observed'] = True
    online['transaction_reference'] = online.get('transaction_reference', online.event_id)
    online = online.rename(columns={'payout_amount_common': 'payout_amount_normalized'})

    sports = sports_betting.copy()
    sports['source_account_id'] = sports.sports_account_id
    sports['source_player_id'] = sports.player_id
    sports['source_event_id'] = sports.event_id
    sports['payout_amount_normalized'] = sports.settlement_amount
    sports['observed'] = True
    sports['transaction_reference'] = sports.get('transaction_reference', sports.event_id)

    land = land_based.copy()
    land['source_account_id'] = _source_accounts(land)
    land['source_player_id'] = land.patron_id
    land['source_event_id'] = land.event_id
    land['event_type'] = land.transaction_type
    land['deposit_amount'] = land.buy_in_amount
    land['withdrawal_amount'] = land.cash_out_amount
    land['wager_amount'] = land.wager_observed_amount
    land['payout_amount_normalized'] = land.payout_observed_amount
    land['region'] = land.property_region
    land['observed'] = land.wager_observation_available.astype(bool)
    land['transaction_reference'] = land.get('transaction_reference', land.event_id)
    unobserved_financial = (~land.observed) & land.event_type.isin(
        ['wager_observation', 'payout_observation']
    )
    land.loc[unobserved_financial, ['wager_amount', 'payout_amount_normalized', 'net_loss']] = np.nan

    normalized = []
    for frame in (online, sports, land):
        attached = _attach_identity(frame, identity_map)
        attached['payout_amount'] = attached.payout_amount_normalized
        attached['amount'] = np.select(
            [
                attached.event_type.isin(['deposit', 'buy_in']),
                attached.event_type.isin(['withdrawal', 'cash_out']),
                attached.event_type.isin(['wager', 'wager_observation']),
                attached.event_type.isin(['payout', 'settlement', 'payout_observation']),
            ],
            [
                attached.deposit_amount,
                attached.withdrawal_amount,
                attached.wager_amount,
                attached.payout_amount,
            ],
            default=np.nan,
        )
        normalized.append(attached[list(COMMON_EVENT_COLUMNS)])
    events = pd.concat(normalized, ignore_index=True)
    events['timestamp'] = pd.to_datetime(events.timestamp)
    return events.sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def deduplicate_events(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove repeated source representations without collapsing transfer legs."""
    frame = events.copy()
    frame['_financial_direction'] = np.select(
        [frame.deposit_amount.gt(0), frame.withdrawal_amount.gt(0),
         frame.wager_amount.gt(0), frame.payout_amount.gt(0)],
        ['deposit', 'withdrawal', 'wager', 'payout'],
        default=frame.event_type.astype(str),
    )
    frame['_amount_key'] = frame.amount.round(2)
    frame['_known_fields'] = frame.notna().sum(axis=1)
    frame = frame.sort_values(
        ['observed', 'identity_confidence', '_known_fields'], ascending=[False, False, False]
    )
    source_duplicate = frame.duplicated(['channel', 'source_event_id'], keep='first')
    representation_duplicate = frame.duplicated(
        ['transaction_reference', '_financial_direction', '_amount_key'], keep='first'
    ) & frame.transaction_reference.notna()
    duplicate = source_duplicate | representation_duplicate
    removed = int(duplicate.sum())
    kept = frame.loc[~duplicate, list(COMMON_EVENT_COLUMNS)]
    return kept.sort_values(['timestamp', 'event_id']).reset_index(drop=True), removed


def _sum_known(series: pd.Series):
    return series.sum(min_count=1)


def activity_summaries(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return source-channel and unified entity summaries without filling gaps with zero."""
    observed = events[events.observed].copy()
    grouping = [
        'activity_entity_id', 'canonical_player_id', 'proposed_canonical_player_id',
        'identity_status', 'channel',
    ]
    channel = observed.groupby(grouping, dropna=False).agg(
        first_observed=('timestamp', 'min'),
        last_observed=('timestamp', 'max'),
        observed_events=('event_id', 'count'),
        observed_deposits=('deposit_amount', _sum_known),
        observed_withdrawals=('withdrawal_amount', _sum_known),
        observed_wagers=('wager_amount', _sum_known),
        observed_payouts=('payout_amount', _sum_known),
        observed_net_losses=('net_loss', _sum_known),
    ).reset_index()
    unified_group = grouping[:-1]
    unified = channel.groupby(unified_group, dropna=False).agg(
        first_observed=('first_observed', 'min'),
        last_observed=('last_observed', 'max'),
        observed_events=('observed_events', 'sum'),
        total_observed_deposits=('observed_deposits', _sum_known),
        total_observed_withdrawals=('observed_withdrawals', _sum_known),
        total_observed_wagers=('observed_wagers', _sum_known),
        total_observed_payouts=('observed_payouts', _sum_known),
        observed_net_losses=('observed_net_losses', _sum_known),
        active_channels=('channel', 'nunique'),
        largest_single_channel_deposits=('observed_deposits', 'max'),
    ).reset_index()
    unified['cross_channel_deposit_uplift'] = (
        unified.total_observed_deposits / unified.largest_single_channel_deposits.replace(0, np.nan)
    )
    return channel, unified


def completeness_report(
    events: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Describe known information gaps; absence is not converted into zero activity."""
    frame = events.copy()
    frame['period'] = frame.timestamp.dt.to_period('W')
    frame['missing_transaction_fields'] = frame[['amount', 'region']].isna().sum(axis=1)
    frame['unlinked_record'] = frame.identity_status.eq('unlinked')
    frame['uncertain_identity_record'] = frame.identity_status.isin(['probable', 'conflicting'])
    frame['unobserved_period'] = ~frame.observed
    report = frame.groupby('activity_entity_id', dropna=False).agg(
        canonical_player_id=('canonical_player_id', 'first'),
        proposed_canonical_player_id=('proposed_canonical_player_id', 'first'),
        identity_status=('identity_status', 'first'),
        identity_confidence=('identity_confidence', 'min'),
        observed_channels=('channel', lambda x: x[frame.loc[x.index, 'observed']].nunique()),
        channels_with_records=('channel', 'nunique'),
        missing_transaction_fields=('missing_transaction_fields', 'sum'),
        known_unobserved_periods=('period', lambda x: x[frame.loc[x.index, 'unobserved_period']].nunique()),
        unlinked_records=('unlinked_record', 'sum'),
        uncertain_identity_records=('uncertain_identity_record', 'sum'),
    ).reset_index()
    report['channel_coverage'] = report.observed_channels / len(CHANNELS)
    report['coverage_start'] = pd.Timestamp(start_time)
    report['coverage_end'] = pd.Timestamp(end_time)
    return report


def behavioral_indicators(events: pd.DataFrame, recent_days: int = 28) -> pd.DataFrame:
    """Compute observed cross-channel changes without imputing missing periods as zero."""
    observed = events[events.observed].copy()
    if observed.empty:
        return pd.DataFrame()
    cutoff = observed.timestamp.max().normalize() - pd.to_timedelta(recent_days - 1, unit='D')
    observed['window'] = np.where(observed.timestamp >= cutoff, 'recent', 'historical')
    observed['week'] = observed.timestamp.dt.to_period('W')
    weekly = observed.groupby(
        ['activity_entity_id', 'channel', 'window', 'week'], dropna=False
    ).agg(deposits=('deposit_amount', _sum_known)).reset_index()
    channel_windows = weekly.groupby(
        ['activity_entity_id', 'channel', 'window'], dropna=False
    ).agg(weekly_deposits=('deposits', 'mean'), observed_weeks=('week', 'nunique')).reset_index()
    values = channel_windows.pivot_table(
        index=['activity_entity_id', 'channel'], columns='window',
        values=['weekly_deposits', 'observed_weeks'], aggfunc='first', dropna=False,
    )
    values.columns = ['_'.join(col) for col in values.columns]
    values = values.reset_index()
    for column in ('weekly_deposits_historical', 'weekly_deposits_recent',
                   'observed_weeks_historical', 'observed_weeks_recent'):
        if column not in values:
            values[column] = np.nan
    enough = values.observed_weeks_historical.ge(2) & values.observed_weeks_recent.ge(2)
    values['channel_change_ratio'] = np.where(
        enough,
        values.weekly_deposits_recent / values.weekly_deposits_historical.replace(0, np.nan),
        np.nan,
    )

    total_weekly = observed.groupby(['activity_entity_id', 'window', 'week']).agg(
        deposits=('deposit_amount', _sum_known), channels=('channel', 'nunique')
    ).reset_index()
    total_windows = total_weekly.groupby(['activity_entity_id', 'window']).agg(
        weekly_deposits=('deposits', 'mean'), observed_weeks=('week', 'nunique'),
        active_channels=('channels', 'max'),
    ).reset_index()
    unified = total_windows.pivot_table(
        index='activity_entity_id', columns='window',
        values=['weekly_deposits', 'observed_weeks', 'active_channels'],
        aggfunc='first', dropna=False,
    )
    unified.columns = ['_'.join(col) for col in unified.columns]
    unified = unified.reset_index()
    for column in ('weekly_deposits_historical', 'weekly_deposits_recent',
                   'observed_weeks_historical', 'observed_weeks_recent',
                   'active_channels_historical', 'active_channels_recent'):
        if column not in unified:
            unified[column] = np.nan
    enough = unified.observed_weeks_historical.ge(2) & unified.observed_weeks_recent.ge(2)
    unified['unified_change_ratio'] = np.where(
        enough,
        unified.weekly_deposits_recent / unified.weekly_deposits_historical.replace(0, np.nan),
        np.nan,
    )
    max_channel = values.groupby('activity_entity_id').channel_change_ratio.max().rename(
        'largest_channel_change_ratio'
    )
    unified = unified.merge(max_channel, on='activity_entity_id', how='left')
    unified['new_channels_recent'] = (
        unified.active_channels_recent - unified.active_channels_historical
    ).clip(lower=0)
    unified['cross_channel_change_difference'] = (
        unified.unified_change_ratio - unified.largest_channel_change_ratio
    )
    unified['sudden_cross_channel_change'] = (
        unified.unified_change_ratio.ge(1.8)
        & (unified.new_channels_recent.ge(1) | unified.cross_channel_change_difference.ge(.4))
    )
    return unified


def integrate_channels(
    players: pd.DataFrame,
    online_casino: pd.DataFrame,
    sports_betting: pd.DataFrame,
    land_based: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    identity_evidence: pd.DataFrame | None = None,
) -> CrossChannelResult:
    """Run normalization, conservative resolution, deduplication, and summaries."""
    evidence = identity_evidence if identity_evidence is not None else build_identity_evidence(
        online_casino, sports_betting, land_based
    )
    identities = resolve_identities(evidence, players.player_id)
    normalized = normalize_channel_events(online_casino, sports_betting, land_based, identities)
    events, removed = deduplicate_events(normalized)
    channel, unified = activity_summaries(events)
    completeness = completeness_report(events, start_time, end_time)
    indicators = behavioral_indicators(events)
    return CrossChannelResult(
        identity_map=identities,
        events=events,
        channel_summaries=channel,
        unified_summaries=unified,
        completeness=completeness,
        indicators=indicators,
        duplicate_records_removed=removed,
    )
