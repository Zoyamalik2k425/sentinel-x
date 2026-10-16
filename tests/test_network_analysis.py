import pandas as pd
import pytest

from network_analysis import build_relationship_graph


def _event(
    event_id,
    account,
    timestamp,
    channel='online_casino',
    player=None,
    identity_status='confirmed',
    proposed=None,
    device=None,
    payment=None,
):
    return {
        'event_id': event_id,
        'source_account_id': account,
        'channel': channel,
        'timestamp': pd.Timestamp(timestamp),
        'observed': True,
        'canonical_player_id': player if identity_status == 'confirmed' else None,
        'proposed_canonical_player_id': proposed,
        'identity_status': identity_status,
        'identity_confidence': 1.0 if identity_status == 'confirmed' else .75,
        'device_id': device,
        'payment_instrument_id': payment,
    }


def test_unrelated_accounts_remain_separate_components():
    events = pd.DataFrame([
        _event('E1', 'A1', '2026-01-01 10:00', player='P1', device='D1', payment='T1'),
        _event('E2', 'A2', '2026-01-05 10:00', player='P2', device='D2', payment='T2'),
    ])
    result = build_relationship_graph(events)

    assert result.statistics['nodes'] == 2
    assert result.statistics['edges'] == 0
    assert result.statistics['connected_components'] == 2
    assert result.statistics['isolated_accounts'] == 2


def test_legitimate_shared_device_is_explained_but_not_declared_unusual():
    events = pd.DataFrame([
        _event('E1', 'A1', '2026-01-01 10:00', player='P1', device='HOUSEHOLD-D'),
        _event('E2', 'A2', '2026-01-02 12:00', player='P2', device='HOUSEHOLD-D'),
    ])
    result = build_relationship_graph(events)
    edge = result.edges.iloc[0]

    assert edge.relationship_types == ('shared_device',)
    assert not result.components.iloc[0].potentially_unusual
    assert 'not proof of fraud' in edge.explanation


def test_strong_multi_evidence_cluster_is_marked_for_investigation():
    rows = []
    for account_number in range(3):
        for occurrence, timestamp in enumerate([
            '2026-03-01 10:00', '2026-03-02 10:00', '2026-03-03 10:00',
        ]):
            rows.append(_event(
                f'E-{account_number}-{occurrence}', f'A{account_number}', timestamp,
                player=f'P{account_number}', device='SHARED-DEVICE', payment='SHARED-TOKEN',
            ))
    result = build_relationship_graph(pd.DataFrame(rows))

    assert result.statistics['largest_component'] == 3
    assert result.statistics['potentially_unusual_components'] == 1
    assert result.components.iloc[0].ring_status == 'Ring flagged'
    assert len(result.edges) == 3
    assert result.edges.correlated_activity_windows.eq(3).all()
    assert result.edges.relationship_types.apply(
        lambda value: {'shared_device', 'shared_payment_instrument', 'correlated_activity'}.issubset(value)
    ).all()
    assert result.nodes.degree.eq(2).all()
    assert result.nodes.weighted_degree.gt(0).all()
    assert result.nodes.investigation_weighted_degree.gt(0).all()
    assert result.nodes.maximum_investigation_strength.gt(0).all()
    assert result.nodes.display_name.str.startswith('Player ').all()


def test_uncertain_identity_matches_are_not_ownership_edges():
    events = pd.DataFrame([
        _event('E1', 'LAND-1', '2026-01-01', channel='land_casino',
               identity_status='probable', proposed='P1'),
        _event('E2', 'ONLINE-1', '2026-01-05', channel='online_casino',
               player='P1'),
    ])
    result = build_relationship_graph(events)

    assert result.statistics['edges'] == 0
    assert result.statistics['connected_components'] == 2


def test_confirmed_cross_channel_accounts_receive_ownership_relationship():
    events = pd.DataFrame([
        _event('E1', 'OC-1', '2026-01-01', channel='online_casino', player='P1'),
        _event('E2', 'SB-1', '2026-01-05', channel='sportsbook', player='P1'),
    ])
    result = build_relationship_graph(events)

    assert result.edges.iloc[0].confirmed_ownership
    assert 'confirmed_cross_channel_ownership' in result.edges.iloc[0].relationship_types
    assert 'confirmed synthetic cross-channel ownership' in result.edges.iloc[0].explanation
    assert result.nodes.investigation_weighted_degree.eq(0).all()


def test_final_risk_scores_are_rejected_as_network_inputs():
    events = pd.DataFrame([
        _event('E1', 'A1', '2026-01-01', player='P1'),
    ]).assign(anomaly_severity=99)

    with pytest.raises(ValueError, match='cannot enter network analysis'):
        build_relationship_graph(events)


def test_plotly_frames_contain_coordinates_and_explanations():
    events = pd.DataFrame([
        _event('E1', 'A1', '2026-01-01', player='P1', payment='T1'),
        _event('E2', 'A2', '2026-01-02', player='P2', payment='T1'),
    ])
    result = build_relationship_graph(events)

    assert {'x', 'y', 'hover_text'}.issubset(result.plotly_nodes.columns)
    assert {'x0', 'y0', 'x1', 'y1', 'hover_text'}.issubset(result.plotly_edges.columns)


def test_plotly_layout_can_be_capped_without_changing_graph_metrics():
    events = pd.DataFrame([
        _event(f'E{number}', f'A{number}', f'2026-01-{number + 1:02} 10:00',
               player=f'P{number}', device=f'D{number}')
        for number in range(5)
    ])
    result = build_relationship_graph(events, maximum_plot_nodes=3)

    assert result.statistics['nodes'] == 5
    assert len(result.nodes) == 5
    assert len(result.plotly_nodes) == 3
