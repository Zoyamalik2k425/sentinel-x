import numpy as np
import pandas as pd

from cross_channel import (
    COMMON_EVENT_COLUMNS,
    activity_summaries,
    behavioral_indicators,
    completeness_report,
    deduplicate_events,
    integrate_channels,
    resolve_identities,
)
from data_gen import generate_dataset


def _event(event_id, player, channel, timestamp, deposit, observed=True, status='confirmed'):
    canonical = player if status == 'confirmed' else None
    account = f'{channel}-{player}'
    return {
        'event_id': event_id,
        'source_event_id': event_id,
        'channel': channel,
        'timestamp': pd.Timestamp(timestamp),
        'event_type': 'deposit',
        'source_account_id': account,
        'source_player_id': player,
        'canonical_player_id': canonical,
        'proposed_canonical_player_id': player if status == 'probable' else None,
        'activity_entity_id': f'PLAYER:{player}' if status == 'confirmed' else f'SOURCE:{account}',
        'identity_status': status,
        'identity_confidence': 1.0 if status == 'confirmed' else .75,
        'deposit_amount': deposit,
        'withdrawal_amount': 0.0,
        'wager_amount': np.nan if not observed else 0.0,
        'payout_amount': 0.0,
        'net_loss': np.nan if not observed else 0.0,
        'amount': deposit if observed else np.nan,
        'region': 'GTA',
        'observed': observed,
        'transaction_reference': event_id,
    }


def test_joint_behavior_is_concerning_when_channels_are_viewed_together():
    rows = []
    for week in range(4):
        rows.append(_event(f'H{week}', 'P-RISK', 'online_casino', f'2026-01-{2 + 7*week:02d}', 100))
    for week in range(4):
        day = 3 + 7 * week
        rows.extend([
            _event(f'RC{week}', 'P-RISK', 'online_casino', f'2026-03-{day:02d}', 100),
            _event(f'RS{week}', 'P-RISK', 'sportsbook', f'2026-03-{day:02d}', 90),
            _event(f'RL{week}', 'P-RISK', 'land_casino', f'2026-03-{day:02d}', 80),
        ])
    events = pd.DataFrame(rows, columns=COMMON_EVENT_COLUMNS)

    indicators = behavioral_indicators(events).iloc[0]
    assert indicators.largest_channel_change_ratio == 1.0
    assert indicators.unified_change_ratio > 2.5
    assert indicators.new_channels_recent == 2
    assert bool(indicators.sudden_cross_channel_change)


def test_incomplete_land_history_remains_missing_not_zero():
    event = _event('L1', 'P-LAND', 'land_casino', '2026-03-01', np.nan, observed=False)
    events = pd.DataFrame([event], columns=COMMON_EVENT_COLUMNS)

    channel, unified = activity_summaries(events)
    completeness = completeness_report(events, pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30'))
    assert channel.empty and unified.empty
    assert completeness.iloc[0].known_unobserved_periods == 1
    assert completeness.iloc[0].missing_transaction_fields >= 1


def test_probable_identity_is_not_automatically_merged():
    evidence = pd.DataFrame([{
        'channel': 'land_casino', 'source_account_id': 'LOY-7',
        'asserted_player_id': 'P000007', 'synthetic_match_confidence': .78,
        'evidence': 'partial loyalty evidence', 'conflicting_identifier': False,
    }])
    resolved = resolve_identities(evidence, ['P000007']).iloc[0]

    assert resolved.identity_status == 'probable'
    assert pd.isna(resolved.canonical_player_id)
    assert resolved.proposed_canonical_player_id == 'P000007'
    assert resolved.identity_confidence == .78


def test_player_with_one_channel_has_partial_not_zero_coverage():
    events = pd.DataFrame([
        _event('O1', 'P-ONLY', 'online_casino', '2026-02-01', 125),
    ], columns=COMMON_EVENT_COLUMNS)
    _, unified = activity_summaries(events)
    completeness = completeness_report(events, pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30'))

    assert unified.iloc[0].active_channels == 1
    assert unified.iloc[0].total_observed_deposits == 125
    assert completeness.iloc[0].channel_coverage == 1 / 3


def test_conflicting_account_identifiers_are_quarantined():
    evidence = pd.DataFrame([
        {'channel': 'sportsbook', 'source_account_id': 'A-9', 'asserted_player_id': 'P1',
         'synthetic_match_confidence': .99},
        {'channel': 'sportsbook', 'source_account_id': 'A-9', 'asserted_player_id': 'P2',
         'synthetic_match_confidence': .99},
    ])
    resolved = resolve_identities(evidence, ['P1', 'P2']).iloc[0]

    assert resolved.identity_status == 'conflicting'
    assert pd.isna(resolved.canonical_player_id)
    assert pd.isna(resolved.proposed_canonical_player_id)


def test_duplicate_source_representations_are_counted_once():
    first = _event('D1', 'P1', 'online_casino', '2026-02-01', 100)
    second = _event('D2', 'P1', 'sportsbook', '2026-02-01', 100)
    first['transaction_reference'] = second['transaction_reference'] = 'TRANSFER-10'
    events = pd.DataFrame([first, second], columns=COMMON_EVENT_COLUMNS)

    deduplicated, removed = deduplicate_events(events)
    assert removed == 1
    assert len(deduplicated) == 1


def test_generated_channel_integration_smoke():
    data = generate_dataset(n_players=80, history_days=70, target_events=2_000, seed=19)
    result = integrate_channels(
        data.players, data.online_casino, data.sports_betting, data.land_based,
        data.start_time, data.end_time,
    )

    assert len(result.events) <= data.event_count
    assert set(result.events.channel.unique()) == {'online_casino', 'sportsbook', 'land_casino'}
    assert result.identity_map.identity_status.isin(
        ['confirmed', 'probable', 'unlinked', 'conflicting']
    ).all()
    probable = result.identity_map[result.identity_map.identity_status == 'probable']
    assert probable.canonical_player_id.isna().all()
    unavailable = result.events[
        (~result.events.observed)
        & result.events.event_type.isin(['wager_observation', 'payout_observation'])
    ]
    assert unavailable.amount.isna().all()
    assert unavailable.net_loss.isna().all()
