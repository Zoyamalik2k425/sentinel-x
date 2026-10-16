import numpy as np
import pandas as pd
import pytest

from cross_channel import COMMON_EVENT_COLUMNS, integrate_channels
from data_gen import generate_dataset
from feature_engineering import (
    GROUND_TRUTH_COLUMNS,
    build_features,
    fairness_audit_attributes,
)


def _profile(player_id, preference='casino|sports', age='30-44', region='GTA', opened='2025-01-01'):
    return {
        'player_id': player_id,
        'account_open_date': pd.Timestamp(opened),
        'gaming_preferences': preference,
        'age_band': age,
        'region': region,
    }


def _event(
    event_id,
    player_id,
    timestamp,
    channel='online_casino',
    event_type='deposit',
    amount=100.0,
    observed=True,
    device='DEV-1',
    payment='PAY-1',
):
    deposit = amount if event_type in ('deposit', 'buy_in') else 0.0
    withdrawal = amount if event_type in ('withdrawal', 'cash_out') else 0.0
    wager = amount if event_type in ('wager', 'wager_observation') else 0.0
    payout = amount if event_type in ('payout', 'settlement', 'payout_observation') else 0.0
    net_loss = wager - payout
    if not observed:
        wager = payout = net_loss = np.nan
        event_amount = np.nan
    else:
        event_amount = amount
    account = f'{channel}-{player_id}'
    return {
        'event_id': event_id,
        'source_event_id': event_id,
        'channel': channel,
        'timestamp': pd.Timestamp(timestamp),
        'event_type': event_type,
        'source_account_id': account,
        'source_player_id': player_id,
        'canonical_player_id': player_id,
        'proposed_canonical_player_id': None,
        'activity_entity_id': f'PLAYER:{player_id}',
        'identity_status': 'confirmed',
        'identity_confidence': 1.0,
        'deposit_amount': deposit,
        'withdrawal_amount': withdrawal,
        'wager_amount': wager,
        'payout_amount': payout,
        'net_loss': net_loss,
        'amount': event_amount,
        'region': 'GTA',
        'observed': observed,
        'transaction_reference': event_id,
        'session_id': f'SESSION-{event_id}',
        'device_id': device,
        'payment_instrument_id': payment,
    }


def _frame(rows):
    return pd.DataFrame(rows, columns=COMMON_EVENT_COLUMNS)


def test_future_transactions_do_not_change_earlier_features():
    profiles = pd.DataFrame([_profile('P1')])
    initial = _frame([
        _event('E1', 'P1', '2026-01-01 10:00', amount=40),
        _event('E2', 'P1', '2026-01-08 10:00', amount=60),
        _event('E3', 'P1', '2026-01-15 10:00', amount=80),
    ])
    future = _event(
        'FUTURE', 'P1', '2026-06-30 23:00', channel='sportsbook',
        amount=1_000_000, device='FUTURE-DEVICE', payment='FUTURE-PAYMENT',
    )

    before = build_features(initial, profiles, minimum_peer_entities=1)
    after = build_features(pd.concat([initial, _frame([future])], ignore_index=True), profiles,
                           minimum_peer_entities=1)
    before = before.set_index('event_id').sort_index()
    after = after[after.event_id.isin(before.index)].set_index('event_id').sort_index()

    pd.testing.assert_frame_equal(before, after, check_dtype=False, check_like=True)


def test_channel_and_unified_features_capture_fragmented_activity():
    profiles = pd.DataFrame([_profile('P1', preference='all_channels')])
    events = _frame([
        _event('C1', 'P1', '2026-03-01', 'online_casino', amount=100),
        _event('S1', 'P1', '2026-03-02', 'sportsbook', amount=90),
        _event('L1', 'P1', '2026-03-03', 'land_casino', event_type='buy_in', amount=80),
    ])

    features = build_features(events, profiles)
    latest = features.loc[features.event_id == 'L1'].iloc[0]
    assert latest.channel_deposit_amount_30d == 80
    assert latest.unified_deposit_amount_30d == 270
    assert latest.active_channels_30d == 3
    assert latest.channel_coverage_30d == 1.0
    assert latest.channel_unified_deposit_difference_30d == 190


def test_incomplete_land_history_preserves_unknown_losses():
    profiles = pd.DataFrame([_profile('P1', preference='land')])
    events = _frame([
        _event('L1', 'P1', '2026-03-01', 'land_casino', 'buy_in', 100),
        _event('L2', 'P1', '2026-03-02', 'land_casino', 'wager_observation', 50, observed=False),
    ])

    latest = build_features(events, profiles).loc[lambda x: x.event_id == 'L2'].iloc[0]
    assert pd.isna(latest.channel_net_loss_30d)
    assert not bool(latest.channel_net_loss_reliable_30d)
    assert latest.known_unobserved_records_30d == 1
    assert latest.observed_record_fraction_30d == .5


def test_new_player_and_zero_mad_are_handled_without_infinity():
    profiles = pd.DataFrame([_profile('P1', opened='2026-04-01')])
    events = _frame([
        _event('E1', 'P1', '2026-04-02', amount=50),
        _event('E2', 'P1', '2026-04-03', amount=50),
        _event('E3', 'P1', '2026-04-04', amount=50),
    ])

    latest = build_features(events, profiles, minimum_peer_entities=1).iloc[-1]
    assert bool(latest.limited_history)
    assert latest.channel_historical_spending_mad == 0
    assert bool(latest.channel_near_zero_mad)
    assert latest.channel_behavioral_stability == 1.0
    numeric = pd.to_numeric(latest, errors='coerce')
    assert not np.isinf(numeric.dropna()).any()


def test_behavioral_peers_exclude_age_and_region_from_model_features():
    profiles = pd.DataFrame([
        _profile('P1', preference='sports', age='19-29', region='North'),
        _profile('P2', preference='sports', age='60+', region='Ottawa'),
    ])
    events = _frame([
        _event('P1-E', 'P1', '2026-03-01', 'sportsbook', amount=100),
        _event('P2-E', 'P2', '2026-03-01', 'sportsbook', amount=200),
    ])

    features = build_features(events, profiles, minimum_peer_entities=2)
    audit = fairness_audit_attributes(profiles)
    assert set(features.behavioral_peer_group) == {'sports'}
    assert (features.peer_entity_count == 2).all()
    assert {'age_band', 'region'}.isdisjoint(features.columns)
    assert {'player_id', 'age_band', 'region'} == set(audit.columns)


def test_ground_truth_inputs_are_rejected():
    profiles = pd.DataFrame([_profile('P1')]).assign(ground_truth_label=True)
    events = _frame([_event('E1', 'P1', '2026-01-01')])

    with pytest.raises(ValueError, match='Ground-truth columns'):
        build_features(events, profiles)


def test_generated_cross_channel_events_produce_documented_features():
    data = generate_dataset(n_players=50, history_days=70, target_events=1_200, seed=17)
    integrated = integrate_channels(
        data.players, data.online_casino, data.sports_betting, data.land_based,
        data.start_time, data.end_time,
    )
    features = build_features(integrated.events, data.players)

    assert len(features) == len(integrated.events)
    assert features.event_id.is_unique
    assert GROUND_TRUTH_COLUMNS.isdisjoint(features.columns)
    assert 'feature_documentation' in features.attrs
