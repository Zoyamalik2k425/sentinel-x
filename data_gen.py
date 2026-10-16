"""Reproducible fictional gaming data for the Sentinel X demonstration.

The schemas in this module are synthetic design choices. They do not represent
or claim knowledge of OLG's internal databases, products, or controls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from configuration import DEFAULT_DATA_CONFIG


CHANNELS = ('online_casino', 'sportsbook', 'land_casino')
REGIONS = ('GTA', 'Ottawa', 'North', 'Southwest', 'East')
SCENARIOS = (
    'sudden_spending_escalation',
    'increasing_losses_and_deposits',
    'unusual_geographic_activity',
    'coordinated_account_relationships',
    'rapid_funds_limited_play',
    'distributed_cross_channel_activity',
    'abrupt_behavioral_change',
    'legitimate_high_spender',
)
RISK_CATEGORY = {
    'sudden_spending_escalation': 'safer_gambling',
    'increasing_losses_and_deposits': 'safer_gambling',
    'unusual_geographic_activity': 'aml_fraud',
    'coordinated_account_relationships': 'aml_fraud',
    'rapid_funds_limited_play': 'aml_fraud',
    'distributed_cross_channel_activity': 'both',
    'abrupt_behavioral_change': 'both',
    'legitimate_high_spender': 'false_positive_control',
}
LABEL_COLUMNS = (
    'player_id', 'scenario_type', 'start_time', 'end_time',
    'target_risk_category', 'ground_truth_label',
)


@dataclass(frozen=True)
class SyntheticGamingData:
    """Channel-specific tables plus labels kept outside operational data."""

    players: pd.DataFrame
    online_casino: pd.DataFrame
    sports_betting: pd.DataFrame
    land_based: pd.DataFrame
    scenario_labels: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    seed: int

    @property
    def event_count(self) -> int:
        return len(self.online_casino) + len(self.sports_betting) + len(self.land_based)


def _money(value: float) -> float:
    return round(max(float(value), 0.0), 2)


def _profiles(rng: np.random.Generator, n_players: int, start: pd.Timestamp):
    age_bands = rng.choice(['19-29', '30-44', '45-59', '60+'], n_players, p=[.23, .34, .27, .16])
    regions = rng.choice(REGIONS, n_players, p=[.44, .14, .12, .16, .14])
    incomes = rng.choice(['under_40k', '40k_70k', '70k_100k', '100k_150k', '150k_plus'],
                         n_players, p=[.20, .30, .24, .17, .09])
    occupations = rng.choice(
        ['service', 'trades', 'professional', 'management', 'student', 'retired', 'not_disclosed'],
        n_players, p=[.20, .16, .25, .12, .08, .12, .07],
    )
    preferences = rng.choice(
        ['casino', 'sports', 'land', 'casino|sports', 'casino|land', 'sports|land', 'all_channels'],
        n_players, p=[.22, .19, .10, .20, .10, .08, .11],
    )
    open_offsets = rng.integers(-1_100, 31, n_players)
    statuses = rng.choice(['active', 'restricted', 'closed'], n_players, p=[.95, .03, .02])
    players = pd.DataFrame({
        'player_id': [f'P{i + 1:06d}' for i in range(n_players)],
        'age_band': age_bands,
        'region': regions,
        'account_open_date': (start + pd.to_timedelta(open_offsets, unit='D')).normalize(),
        'income_band': incomes,
        'occupation_category': occupations,
        'gaming_preferences': preferences,
        'account_status': statuses,
    })

    income_midpoint = {
        'under_40k': 32_000, '40k_70k': 55_000, '70k_100k': 85_000,
        '100k_150k': 125_000, '150k_plus': 190_000,
    }
    annual_income = np.array([income_midpoint[x] for x in incomes])
    weekly_budget = np.clip(annual_income / 52 * rng.lognormal(-2.75, .52, n_players), 12, 1_500)
    activity = rng.lognormal(0, .72, n_players)
    return players, weekly_budget, activity


def _scenario_table(
    rng: np.random.Generator,
    players: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fraction: float,
) -> pd.DataFrame:
    count = int(round(len(players) * fraction))
    chosen = rng.permutation(players.player_id.to_numpy())[:count]
    rows = []
    for i, player_id in enumerate(chosen):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        scenario_start = start + pd.to_timedelta(int(rng.integers(105, 151)), unit='D')
        rows.append({
            'player_id': player_id,
            'scenario_type': scenario,
            'start_time': scenario_start,
            'end_time': end,
            'target_risk_category': RISK_CATEGORY[scenario],
            'ground_truth_label': scenario != 'legitimate_high_spender',
        })
    return pd.DataFrame(rows, columns=LABEL_COLUMNS).sort_values('player_id').reset_index(drop=True)


def _preference_multiplier(preference: str, channel: str) -> float:
    token = {'online_casino': 'casino', 'sportsbook': 'sports', 'land_casino': 'land'}[channel]
    return 2.2 if token in preference else .42


def _timestamps(
    rng: np.random.Generator,
    count: int,
    first_day: pd.Timestamp,
    end: pd.Timestamp,
    channel: str,
    scenario: str | None,
    scenario_start: pd.Timestamp | None,
) -> list[pd.Timestamp]:
    if count == 0 or first_day > end:
        return []
    days = pd.date_range(first_day.normalize(), end.normalize(), freq='D')
    weekday = days.dayofweek.to_numpy()
    if channel == 'land_casino':
        weights = np.where(weekday >= 4, 1.75, .72)
        hour_mean = 20.5
    elif channel == 'sportsbook':
        weights = np.where(weekday >= 4, 1.45, .84)
        hour_mean = 18.5
    else:
        weights = np.where(weekday >= 4, 1.20, .93)
        hour_mean = 19.5
    if scenario_start is not None and scenario not in (None, 'legitimate_high_spender'):
        weights = weights * np.where(days >= scenario_start.normalize(), 2.8, 1.0)
    picked = rng.choice(len(days), size=count, replace=True, p=weights / weights.sum())
    hours = np.clip(np.rint(rng.normal(hour_mean, 4.1, count)), 0, 23).astype(int)
    minutes = rng.integers(0, 60, count)
    seconds = rng.integers(0, 60, count)
    values = [days[d] + pd.to_timedelta(int(h) * 3600 + int(m) * 60 + int(s), unit='s')
              for d, h, m, s in zip(picked, hours, minutes, seconds)]
    return sorted(min(value, end) for value in values)


def _scenario_context(scenario: str | None, timestamp: pd.Timestamp, scenario_start: pd.Timestamp | None):
    active = scenario is not None and scenario_start is not None and timestamp >= scenario_start
    if not active or scenario == 'legitimate_high_spender':
        return 1.0, 0.0
    progress = float(np.clip((timestamp - scenario_start).days / 45, 0, 1))
    if scenario == 'sudden_spending_escalation':
        return 4.2, progress
    if scenario == 'increasing_losses_and_deposits':
        return 1.5 + 2.5 * progress, progress
    if scenario in ('distributed_cross_channel_activity', 'abrupt_behavioral_change'):
        return 2.7, progress
    if scenario == 'rapid_funds_limited_play':
        return 3.5, progress
    return 1.35, progress


def _location(rng, home_region, scenario, active):
    if scenario == 'unusual_geographic_activity' and active and rng.random() < .72:
        return str(rng.choice([r for r in REGIONS if r != home_region]))
    return home_region if rng.random() < .96 else str(rng.choice(REGIONS))


def _channel_counts(rng, players, activity, target, channel):
    weights = activity * np.array([
        _preference_multiplier(pref, channel) for pref in players.gaming_preferences
    ])
    return rng.multinomial(target, weights / weights.sum())


def _online_events(rng, players, budgets, counts, start, end, scenarios):
    rows = []
    games = ('slots', 'table_games', 'live_dealer', 'instant_games')
    for idx, (player, count) in enumerate(zip(players.itertuples(index=False), counts)):
        scenario, scenario_start, group = scenarios.get(player.player_id, (None, None, None))
        first = max(start, pd.Timestamp(player.account_open_date))
        times = _timestamps(rng, int(count), first, end, 'online_casino', scenario, scenario_start)
        balance = 0.0
        high = 3.8 if scenario == 'legitimate_high_spender' else 1.0
        base = budgets[idx] * high / 5
        for timestamp in times:
            factor, progress = _scenario_context(scenario, timestamp, scenario_start)
            active = scenario_start is not None and timestamp >= scenario_start
            probs = np.array([.25, .53, .14, .08])  # deposit, wager, payout, withdrawal
            if active and scenario == 'rapid_funds_limited_play':
                probs = np.array([.43, .10, .04, .43])
            elif active and scenario == 'increasing_losses_and_deposits':
                probs = np.array([.37, .50, .08, .05])
            event_type = str(rng.choice(['deposit', 'wager', 'payout', 'withdrawal'], p=probs))
            before = balance
            amount = _money(base * factor * rng.lognormal(-.15, .55))
            if event_type in ('wager', 'withdrawal') and balance < 1:
                event_type = 'deposit'
            if event_type == 'deposit':
                amount = max(amount, 5.0)
                balance += amount
            elif event_type == 'wager':
                play_scale = .08 if active and scenario == 'rapid_funds_limited_play' else rng.uniform(.10, .48)
                amount = _money(min(balance, max(1, balance * play_scale)))
                balance -= amount
            elif event_type == 'payout':
                payout_scale = .32 if active and scenario == 'increasing_losses_and_deposits' else rng.uniform(.35, 1.7)
                amount = _money(base * factor * payout_scale)
                balance += amount
            else:
                amount = _money(min(balance, max(1, balance * rng.uniform(.20, .82))))
                balance -= amount
            device = f'DEV-{player.player_id}-{int(rng.integers(1, 3))}'
            instrument = f'PAY-{player.player_id}'
            if active and scenario == 'coordinated_account_relationships':
                device, instrument = f'SHARED-DEV-{group:03d}', f'SHARED-PAY-{group:03d}'
            if active and scenario == 'abrupt_behavioral_change':
                device = f'NEW-DEV-{player.player_id}'
            rows.append({
                'event_id': f'OC{len(rows) + 1:09d}', 'channel': 'online_casino',
                'player_id': player.player_id, 'account_id': f'OC-ACT-{player.player_id}',
                'timestamp': timestamp, 'event_type': event_type,
                'deposit_amount': amount if event_type == 'deposit' else 0.0,
                'withdrawal_amount': amount if event_type == 'withdrawal' else 0.0,
                'wager_amount': amount if event_type == 'wager' else 0.0,
                'payout_amount': amount if event_type == 'payout' else 0.0,
                'net_loss': amount if event_type == 'wager' else (-amount if event_type == 'payout' else 0.0),
                'balance_before': round(before, 2), 'balance_after': round(balance, 2),
                'session_id': f'OCS-{player.player_id}-{timestamp:%Y%m%d}-{timestamp.hour // 4}',
                'session_duration_minutes': int(np.clip(rng.lognormal(3.5, .65), 3, 360)),
                'device_id': device, 'payment_instrument_id': instrument,
                'game_category': str(rng.choice(games)),
                'region': _location(rng, player.region, scenario, active),
            })
    return pd.DataFrame(rows).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def _sports_events(rng, players, budgets, counts, start, end, scenarios):
    rows = []
    bet_number = 0
    sports = ('hockey', 'basketball', 'football', 'baseball', 'soccer', 'other')
    for idx, (player, count) in enumerate(zip(players.itertuples(index=False), counts)):
        scenario, scenario_start, group = scenarios.get(player.player_id, (None, None, None))
        first = max(start, pd.Timestamp(player.account_open_date))
        times = _timestamps(rng, int(count), first, end, 'sportsbook', scenario, scenario_start)
        balance, open_bets = 0.0, []
        high = 3.8 if scenario == 'legitimate_high_spender' else 1.0
        base = budgets[idx] * high / 6
        for timestamp in times:
            factor, progress = _scenario_context(scenario, timestamp, scenario_start)
            active = scenario_start is not None and timestamp >= scenario_start
            probs = np.array([.25, .43, .24, .08])
            if active and scenario == 'rapid_funds_limited_play':
                probs = np.array([.44, .09, .05, .42])
            elif active and scenario == 'increasing_losses_and_deposits':
                probs = np.array([.37, .46, .13, .04])
            event_type = str(rng.choice(['deposit', 'wager', 'settlement', 'withdrawal'], p=probs))
            if event_type == 'settlement' and not open_bets:
                event_type = 'wager'
            if event_type in ('wager', 'withdrawal') and balance < 1:
                event_type = 'deposit'
            before, bet_id, bet_timestamp = balance, None, pd.NaT
            amount = _money(base * factor * rng.lognormal(-.22, .55))
            deposit = withdrawal = wager = settlement = 0.0
            if event_type == 'deposit':
                deposit = max(amount, 5.0)
                balance += deposit
            elif event_type == 'wager':
                scale = .07 if active and scenario == 'rapid_funds_limited_play' else rng.uniform(.12, .52)
                wager = _money(min(balance, max(1, balance * scale)))
                balance -= wager
                bet_number += 1
                bet_id = f'BET{bet_number:09d}'
                bet_timestamp = timestamp
                open_bets.append((bet_id, wager, timestamp))
            elif event_type == 'settlement':
                bet_id, stake, bet_timestamp = open_bets.pop(0)
                win_probability = .22 if active and scenario == 'increasing_losses_and_deposits' else .42
                settlement = _money(stake * rng.uniform(1.15, 3.2)) if rng.random() < win_probability else 0.0
                balance += settlement
            else:
                withdrawal = _money(min(balance, max(1, balance * rng.uniform(.20, .82))))
                balance -= withdrawal
            instrument = f'SB-PAY-{player.player_id}'
            if active and scenario == 'coordinated_account_relationships':
                instrument = f'SHARED-PAY-{group:03d}'
            rows.append({
                'event_id': f'SB{len(rows) + 1:09d}', 'channel': 'sportsbook',
                'player_id': player.player_id, 'sports_account_id': f'SB-ACT-{player.player_id}',
                'timestamp': timestamp, 'bet_timestamp': bet_timestamp, 'event_type': event_type,
                'deposit_amount': round(deposit, 2), 'wager_amount': round(wager, 2),
                'settlement_amount': round(settlement, 2), 'withdrawal_amount': round(withdrawal, 2),
                'net_loss': round(wager - settlement, 2),
                'balance_before': round(before, 2), 'balance_after': round(balance, 2),
                'bet_id': bet_id, 'sport_category': str(rng.choice(sports)),
                'payment_instrument_id': instrument,
                'device_id': f'SB-DEV-{player.player_id}-{int(rng.integers(1, 3))}',
                'region': _location(rng, player.region, scenario, active),
            })
    return pd.DataFrame(rows).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def _land_events(rng, players, budgets, counts, start, end, scenarios):
    rows = []
    for idx, (player, count) in enumerate(zip(players.itertuples(index=False), counts)):
        scenario, scenario_start, group = scenarios.get(player.player_id, (None, None, None))
        first = max(start, pd.Timestamp(player.account_open_date))
        times = _timestamps(rng, int(count), first, end, 'land_casino', scenario, scenario_start)
        balance = 0.0
        high = 3.8 if scenario == 'legitimate_high_spender' else 1.0
        base = budgets[idx] * high / 4
        for timestamp in times:
            factor, progress = _scenario_context(scenario, timestamp, scenario_start)
            active = scenario_start is not None and timestamp >= scenario_start
            linked = bool(rng.random() < .68)
            event_type = str(rng.choice(
                ['buy_in', 'wager_observation', 'payout_observation', 'cash_out'],
                p=[.31, .39, .18, .12],
            ))
            if event_type in ('wager_observation', 'cash_out') and balance < 1:
                event_type = 'buy_in'
            before = balance
            amount = _money(base * factor * rng.lognormal(-.10, .60))
            buy_in = cash_out = wager = payout = 0.0
            if event_type == 'buy_in':
                buy_in = max(amount, 10.0)
                balance += buy_in
            elif event_type == 'wager_observation':
                wager = _money(min(balance, max(1, balance * rng.uniform(.12, .55))))
                balance -= wager
            elif event_type == 'payout_observation':
                payout = amount
                balance += payout
            else:
                cash_out = _money(min(balance, max(1, balance * rng.uniform(.25, .90))))
                balance -= cash_out
            observed = event_type not in ('wager_observation', 'payout_observation') or rng.random() < .57
            rows.append({
                'event_id': f'LB{len(rows) + 1:09d}', 'channel': 'land_casino',
                'patron_id': player.player_id if linked else None,
                'timestamp': timestamp, 'transaction_type': event_type,
                'buy_in_amount': round(buy_in, 2), 'cash_out_amount': round(cash_out, 2),
                'wager_observed_amount': round(wager, 2), 'payout_observed_amount': round(payout, 2),
                'net_loss': round(wager - payout, 2),
                'balance_before': round(before, 2) if linked else np.nan,
                'balance_after': round(balance, 2) if linked else np.nan,
                'loyalty_id': f'LOY-{player.player_id}' if linked and rng.random() < .74 else None,
                'property_region': _location(rng, player.region, scenario, active),
                'patron_linked': linked,
                'linkage_confidence': round(float(rng.uniform(.82, 1.0)), 3) if linked else 0.0,
                'wager_observation_available': bool(observed),
            })
    return pd.DataFrame(rows).sort_values(['timestamp', 'event_id']).reset_index(drop=True)


def generate_dataset(
    n_players: int = DEFAULT_DATA_CONFIG.n_players,
    history_days: int = DEFAULT_DATA_CONFIG.history_days,
    target_events: int | None = None,
    seed: int = DEFAULT_DATA_CONFIG.seed,
    scenario_fraction: float = DEFAULT_DATA_CONFIG.risky_scenario_fraction,
) -> SyntheticGamingData:
    """Generate a six-month, event-level fictional environment.

    ``target_events`` defaults to 50 events per player (100,000 for the default
    2,000-player population) and must remain in the requested 80k–120k range
    when using the default population.
    """
    if n_players < 1 or history_days < 1:
        raise ValueError('n_players and history_days must be positive')
    if not 0 <= scenario_fraction <= 1:
        raise ValueError('scenario_fraction must be between 0 and 1')
    target_events = int(target_events if target_events is not None else n_players * 50)
    if target_events < n_players:
        raise ValueError('target_events must be at least n_players')

    rng = np.random.default_rng(seed)
    start = pd.Timestamp(DEFAULT_DATA_CONFIG.start_date)
    end = start + pd.to_timedelta(history_days, unit='D') - pd.to_timedelta(1, unit='s')
    players, budgets, activity = _profiles(rng, n_players, start)
    labels = _scenario_table(rng, players, start, end, scenario_fraction)
    label_lookup = {}
    coordinated_count = 0
    for row in labels.itertuples(index=False):
        group = None
        if row.scenario_type == 'coordinated_account_relationships':
            group = coordinated_count // 4
            coordinated_count += 1
        label_lookup[row.player_id] = (row.scenario_type, row.start_time, group)

    casino_target = int(round(target_events * .46))
    sports_target = int(round(target_events * .34))
    land_target = target_events - casino_target - sports_target
    casino_counts = _channel_counts(rng, players, activity, casino_target, 'online_casino')
    sports_counts = _channel_counts(rng, players, activity, sports_target, 'sportsbook')
    land_counts = _channel_counts(rng, players, activity, land_target, 'land_casino')

    return SyntheticGamingData(
        players=players,
        online_casino=_online_events(rng, players, budgets, casino_counts, start, end, label_lookup),
        sports_betting=_sports_events(rng, players, budgets, sports_counts, start, end, label_lookup),
        land_based=_land_events(rng, players, budgets, land_counts, start, end, label_lookup),
        scenario_labels=labels,
        start_time=start,
        end_time=end,
        seed=seed,
    )


def to_operational_events(dataset: SyntheticGamingData) -> pd.DataFrame:
    """Normalize linkable events for the existing operational feature pipeline."""
    casino = dataset.online_casino.assign(
        player_id=lambda x: x.player_id,
        deposit=lambda x: x.deposit_amount,
        wager=lambda x: x.wager_amount,
        payment_token=lambda x: x.payment_instrument_id,
        observed=True,
    )
    sports = dataset.sports_betting.assign(
        player_id=lambda x: x.player_id,
        deposit=lambda x: x.deposit_amount,
        wager=lambda x: x.wager_amount,
        payment_token=lambda x: x.payment_instrument_id,
        observed=True,
    )
    land = dataset.land_based[dataset.land_based.patron_linked].assign(
        player_id=lambda x: x.patron_id,
        deposit=lambda x: x.buy_in_amount,
        wager=lambda x: x.wager_observed_amount,
        payment_token=lambda x: x.loyalty_id,
        device_id=None,
        region=lambda x: x.property_region,
        observed=lambda x: x.wager_observation_available,
    )
    selected = []
    for frame in (casino, sports, land):
        part = frame[['event_id', 'player_id', 'timestamp', 'channel', 'deposit', 'wager',
                      'net_loss', 'region', 'payment_token', 'device_id', 'observed']].copy()
        selected.append(part.rename(columns={'event_id': 'tx_id'}))
    events = pd.concat(selected, ignore_index=True).sort_values(['timestamp', 'tx_id']).reset_index(drop=True)
    events['week'] = ((events.timestamp - dataset.start_time).dt.days // 7).astype(int)
    return events


def generate(
    n_players: int = DEFAULT_DATA_CONFIG.n_players,
    weeks: int | None = None,
    seed: int = DEFAULT_DATA_CONFIG.seed,
    target_events: int | None = None,
):
    """Compatibility API returning profiles, normalized events, and held labels."""
    history_days = weeks * 7 if weeks is not None else DEFAULT_DATA_CONFIG.history_days
    dataset = generate_dataset(n_players, history_days, target_events, seed)
    return dataset.players, to_operational_events(dataset), dataset.scenario_labels


def validate_dataset(dataset: SyntheticGamingData) -> dict[str, bool | int]:
    """Run structural and ledger invariants without consulting model code."""
    event_frames = (dataset.online_casino, dataset.sports_betting, dataset.land_based)
    all_ids = pd.concat([frame.event_id for frame in event_frames], ignore_index=True)
    timestamps_valid = all(
        frame.timestamp.between(dataset.start_time, dataset.end_time).all() for frame in event_frames
    )
    channel_assignments = (
        set(dataset.online_casino.channel) == {'online_casino'}
        and set(dataset.sports_betting.channel) == {'sportsbook'}
        and set(dataset.land_based.channel) == {'land_casino'}
    )
    account_open = dataset.players.set_index('player_id').account_open_date
    linked_land_times = dataset.land_based.dropna(subset=['patron_id'])
    events_after_account_open = (
        (dataset.online_casino.timestamp >= dataset.online_casino.player_id.map(account_open)).all()
        and (dataset.sports_betting.timestamp >= dataset.sports_betting.player_id.map(account_open)).all()
        and (linked_land_times.timestamp >= linked_land_times.patron_id.map(account_open)).all()
    )

    casino_delta = (dataset.online_casino.deposit_amount + dataset.online_casino.payout_amount
                     - dataset.online_casino.withdrawal_amount - dataset.online_casino.wager_amount)
    sports_delta = (dataset.sports_betting.deposit_amount + dataset.sports_betting.settlement_amount
                     - dataset.sports_betting.withdrawal_amount - dataset.sports_betting.wager_amount)
    linked_land = dataset.land_based[dataset.land_based.patron_linked]
    land_delta = (linked_land.buy_in_amount + linked_land.payout_observed_amount
                  - linked_land.cash_out_amount - linked_land.wager_observed_amount)
    balance_consistent = (
        np.allclose(dataset.online_casino.balance_after, dataset.online_casino.balance_before + casino_delta,
                    atol=.011)
        and np.allclose(dataset.sports_betting.balance_after, dataset.sports_betting.balance_before + sports_delta,
                        atol=.011)
        and np.allclose(linked_land.balance_after, linked_land.balance_before + land_delta, atol=.011)
        and (dataset.online_casino.balance_after >= 0).all()
        and (dataset.sports_betting.balance_after >= 0).all()
        and (linked_land.balance_after >= 0).all()
    )
    forbidden = {'scenario_type', 'start_time', 'end_time', 'target_risk_category', 'ground_truth_label'}
    labels_separate = forbidden.isdisjoint(dataset.players.columns) and all(
        forbidden.isdisjoint(frame.columns) for frame in event_frames
    )
    results = {
        'players': len(dataset.players),
        'online_casino_events': len(dataset.online_casino),
        'sports_betting_events': len(dataset.sports_betting),
        'land_based_events': len(dataset.land_based),
        'total_events': dataset.event_count,
        'scenario_players': len(dataset.scenario_labels),
        'unique_player_ids': bool(dataset.players.player_id.is_unique),
        'unique_event_ids': bool(all_ids.is_unique),
        'timestamps_valid': bool(timestamps_valid),
        'events_after_account_open': bool(events_after_account_open),
        'balances_consistent': bool(balance_consistent),
        'channel_assignments_valid': bool(channel_assignments),
        'scenario_labels_separate': bool(labels_separate),
        'unlinked_land_events_present': bool(dataset.land_based.patron_id.isna().any()),
    }
    return results


def save_dataset(dataset: SyntheticGamingData, output_dir: str | Path = 'generated_data') -> dict[str, Path]:
    """Persist deterministic gzip-compressed CSV artifacts and metadata locally."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = {
        'players': dataset.players,
        'online_casino_events': dataset.online_casino,
        'sports_betting_events': dataset.sports_betting,
        'land_based_events': dataset.land_based,
        'scenario_labels': dataset.scenario_labels,
    }
    paths = {}
    for name, frame in frames.items():
        path = destination / f'{name}.csv.gz'
        frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
        paths[name] = path
    metadata = {
        'description': 'Fictional synthetic data; not an OLG internal schema.',
        'seed': dataset.seed,
        'start_time': dataset.start_time.isoformat(),
        'end_time': dataset.end_time.isoformat(),
        'counts': validate_dataset(dataset),
    }
    metadata_path = destination / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    paths['metadata'] = metadata_path
    return paths


def _main():
    parser = argparse.ArgumentParser(description='Generate fictional Sentinel X gaming events.')
    parser.add_argument('--players', type=int, default=DEFAULT_DATA_CONFIG.n_players)
    parser.add_argument('--events', type=int, default=DEFAULT_DATA_CONFIG.target_events)
    parser.add_argument('--days', type=int, default=DEFAULT_DATA_CONFIG.history_days)
    parser.add_argument('--seed', type=int, default=DEFAULT_DATA_CONFIG.seed)
    parser.add_argument('--output', default='generated_data')
    args = parser.parse_args()
    dataset = generate_dataset(args.players, args.days, args.events, args.seed)
    paths = save_dataset(dataset, args.output)
    print(json.dumps(validate_dataset(dataset), indent=2))
    print(f'Saved {len(paths)} artifacts to {Path(args.output).resolve()}')


if __name__ == '__main__':
    _main()
