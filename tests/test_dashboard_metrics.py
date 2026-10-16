import pandas as pd

from dashboard_metrics import (
    evaluate_at_budget,
    fairness_audit,
    performance_by_channel,
    performance_by_scenario,
    ranked_queue,
)


def _detail(value, points):
    return {'available': True, 'signal_value': value, 'score_contribution': points}


def _scored():
    rows = []
    for number, (aml, safer) in enumerate([(90, 10), (20, 80), (70, 30), (10, 5)]):
        player = None if number == 2 else f'P{number}'
        rows.append({
            'player_id': player,
            'activity_entity_id': f'PLAYER:{player}' if player else 'SOURCE:land:L1',
            'aml_fraud_score': aml,
            'safer_gambling_score': safer,
            'risk_band': 'high' if max(aml, safer) >= 60 else 'low',
            'signal_contributions': {
                'aml_fraud': {'transaction_anomaly': _detail(aml / 100, aml / 4)},
                'safer_gambling': {'deposit_escalation': _detail(safer / 100, safer / 4)},
            },
            'data_coverage': {
                'channel_coverage_30d': 2 / 3,
                'data_sufficiency_status': 'moderate',
            },
            'identity_confidence': 1.0 if player else .6,
            'supporting_transaction_ids': [],
        })
    return pd.DataFrame(rows)


def _labels():
    return pd.DataFrame([
        {'player_id': 'P0', 'scenario_type': 'scenario_a', 'ground_truth_label': True},
        {'player_id': 'P1', 'scenario_type': 'scenario_b', 'ground_truth_label': True},
        {'player_id': 'P3', 'scenario_type': 'legitimate_high_spender', 'ground_truth_label': False},
    ])


def test_queue_projection_and_fixed_budget_metrics_are_computed():
    queue = ranked_queue(_scored(), budget=3, case_records={
        'CASE-PLAYER:P0': {'status': 'Under Review'},
    })
    metrics = evaluate_at_budget(queue, _labels(), budget=3)

    assert queue.priority_rank.tolist() == [1, 2, 3, 4]
    assert queue.iloc[0].primary_alert_reason == 'transaction-pattern anomaly'
    assert queue.loc[queue.player_id.eq('P0'), 'display_id'].iloc[0] == 'Player P0'
    assert 'account L1' in queue.loc[queue.player_id.isna(), 'display_id'].iloc[0]
    assert queue.iloc[0].investigation_status == 'Under Review'
    assert metrics['true_positives'] == 2
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0
    assert metrics['unlinked_alerts_not_labelled'] == 1
    assert metrics['precision_at_k'] == 1
    assert metrics['recall_at_k'] == 1


def test_scenario_channel_and_fairness_tables_use_actual_membership():
    queue = ranked_queue(_scored(), budget=3)
    events = pd.DataFrame([
        {'canonical_player_id': 'P0', 'channel': 'online_casino', 'observed': True},
        {'canonical_player_id': 'P1', 'channel': 'sportsbook', 'observed': True},
        {'canonical_player_id': 'P3', 'channel': 'online_casino', 'observed': True},
    ])
    profiles = pd.DataFrame([
        {'player_id': 'P0', 'age_band': '25-34', 'region': 'GTA'},
        {'player_id': 'P1', 'age_band': '35-44', 'region': 'East'},
        {'player_id': 'P3', 'age_band': '25-34', 'region': 'GTA'},
    ])

    scenarios = performance_by_scenario(queue, _labels(), 3)
    channels = performance_by_channel(queue, _labels(), events, 3)
    audit = fairness_audit(queue, profiles, 'age_band', 3)

    assert scenarios.true_positives.sum() == 2
    assert set(audit.sample_status) == {'low sample'}
    assert set(channels.channel) == {'online_casino', 'sportsbook'}
    assert channels.false_negatives.sum() == 0
    assert audit.reviewed.sum() == 2

