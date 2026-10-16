import pandas as pd

from data_gen import (
    LABEL_COLUMNS,
    generate_dataset,
    save_dataset,
    validate_dataset,
)
from signals import GROUND_TRUTH_COLUMNS


def _small_dataset():
    return generate_dataset(n_players=120, history_days=182, target_events=6_000, seed=731)


def test_unique_ids_channels_and_required_counts():
    data = _small_dataset()
    validation = validate_dataset(data)
    event_ids = pd.concat([
        data.online_casino.event_id,
        data.sports_betting.event_id,
        data.land_based.event_id,
    ], ignore_index=True)

    assert data.players.player_id.is_unique
    assert event_ids.is_unique
    assert data.event_count == 6_000
    assert validation['channel_assignments_valid']
    assert set(data.online_casino.channel) == {'online_casino'}
    assert set(data.sports_betting.channel) == {'sportsbook'}
    assert set(data.land_based.channel) == {'land_casino'}
    assert data.land_based.patron_id.isna().any()


def test_timestamps_follow_account_and_dataset_boundaries():
    data = _small_dataset()
    opened = data.players.set_index('player_id').account_open_date

    for frame in (data.online_casino, data.sports_betting):
        assert frame.timestamp.between(data.start_time, data.end_time).all()
        assert (frame.timestamp >= frame.player_id.map(opened)).all()
    linked_land = data.land_based.dropna(subset=['patron_id'])
    assert linked_land.timestamp.between(data.start_time, data.end_time).all()
    assert (linked_land.timestamp >= linked_land.patron_id.map(opened)).all()
    settlements = data.sports_betting[data.sports_betting.event_type == 'settlement']
    assert (settlements.bet_timestamp <= settlements.timestamp).all()


def test_balances_and_financial_flows_are_consistent():
    data = _small_dataset()
    assert validate_dataset(data)['balances_consistent']

    for frame, player_col in (
        (data.online_casino, 'player_id'),
        (data.sports_betting, 'player_id'),
    ):
        ordered = frame.sort_values([player_col, 'timestamp', 'event_id'])
        previous = ordered.groupby(player_col).balance_after.shift()
        comparable = previous.notna()
        assert (ordered.loc[comparable, 'balance_before'] - previous[comparable]).abs().max() < .011
        assert (ordered.balance_after >= 0).all()


def test_generation_is_reproducible():
    first = _small_dataset()
    second = _small_dataset()
    for name in ('players', 'online_casino', 'sports_betting', 'land_based', 'scenario_labels'):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))


def test_scenario_labels_are_separate_and_include_hard_negatives():
    data = _small_dataset()
    operational = (data.players, data.online_casino, data.sports_betting, data.land_based)

    assert tuple(data.scenario_labels.columns) == LABEL_COLUMNS
    malicious = data.scenario_labels[data.scenario_labels.ground_truth_label]
    assert len(malicious) == round(len(data.players) * .03)
    assert (~data.scenario_labels.ground_truth_label).any()
    assert data.scenario_labels.player_id.isin(data.players.player_id).all()
    assert all(GROUND_TRUTH_COLUMNS.isdisjoint(frame.columns) for frame in operational)
    assert validate_dataset(data)['scenario_labels_separate']


def test_dataset_can_be_saved_and_regenerated(tmp_path):
    data = _small_dataset()
    paths = save_dataset(data, tmp_path)

    assert set(paths) == {
        'players', 'online_casino_events', 'sports_betting_events',
        'land_based_events', 'scenario_labels', 'metadata',
    }
    assert all(path.exists() for path in paths.values())
    restored_players = pd.read_csv(paths['players'])
    assert len(restored_players) == len(data.players)
