import pandas as pd

from cross_channel import integrate_channels
from dashboard_features import build_dashboard_feature_history
from data_gen import generate_dataset
from ml.anomaly_detection import DEFAULT_ANOMALY_FEATURES
from ml.peer_analysis import DEFAULT_CLUSTER_FEATURES, DEFAULT_COMPARISON_FEATURES


def _environment():
    data = generate_dataset(n_players=30, history_days=90, target_events=600, seed=17)
    integrated = integrate_channels(
        data.players, data.online_casino, data.sports_betting, data.land_based,
        data.start_time, data.end_time,
    )
    return data, integrated.events


def test_dashboard_snapshots_satisfy_model_contract_and_remain_label_free():
    data, events = _environment()
    features = build_dashboard_feature_history(events, data.players)
    required = (
        set(DEFAULT_ANOMALY_FEATURES)
        | set(DEFAULT_CLUSTER_FEATURES)
        | set(DEFAULT_COMPARISON_FEATURES)
        | {'activity_entity_id', 'event_id', 'timestamp', 'peer_deposit_robust_z'}
    )

    assert required.issubset(features.columns)
    assert 'ground_truth_label' not in features
    assert features.groupby('activity_entity_id').size().max() <= 7
    assert features.timestamp.is_monotonic_increasing


def test_future_records_do_not_change_existing_dashboard_snapshots():
    data, events = _environment()
    cutoffs = list(pd.date_range(events.timestamp.min(), events.timestamp.max(), periods=7))[1:]
    before = build_dashboard_feature_history(events, data.players, snapshot_times=cutoffs)
    future = events.iloc[[0]].copy()
    future['event_id'] = 'FUTURE-EVENT'
    future['source_event_id'] = 'FUTURE-EVENT'
    future['transaction_reference'] = 'FUTURE-EVENT'
    latest = pd.Timestamp(events.timestamp.max().value, unit='ns')
    future['timestamp'] = pd.Timestamp(latest.value + 365 * 86_400_000_000_000, unit='ns')
    future['deposit_amount'] = 1_000_000.0
    after = build_dashboard_feature_history(
        pd.concat([events, future]), data.players, snapshot_times=cutoffs
    )
    columns = ['activity_entity_id', 'timestamp', 'unified_deposit_amount_30d']
    pd.testing.assert_frame_equal(
        before[columns].reset_index(drop=True),
        after[columns].reset_index(drop=True),
        check_dtype=False,
    )
