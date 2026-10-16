import numpy as np
import pandas as pd
import pytest

from scoring import prioritize_investigations, score_risk_indices


def _feature_rows():
    base = {
        'timestamp': pd.Timestamp('2026-06-30'),
        'channel': 'online_casino',
        'active_channels_30d': 1,
        'channel_unified_deposit_difference_30d': 0.0,
        'unified_deposit_amount_30d': 1000.0,
        'channel_participation_change': 0.0,
        'unified_spending_acceleration': 0.0,
        'unified_net_loss_7d': 7.0,
        'unified_net_loss_30d': 30.0,
        'unified_net_loss_reliable_30d': True,
        'unified_session_frequency_change': 0.0,
        'unified_wager_volume_7d': 7.0,
        'unified_wager_volume_30d': 30.0,
        'peer_deposit_robust_z': 0.0,
        'channel_coverage_30d': 1 / 3,
        'observed_record_fraction_30d': 1.0,
        'missing_field_fraction_30d': 0.0,
        'known_unobserved_records_30d': 0,
        'data_sufficiency_status': 'sufficient',
        'identity_confidence': 1.0,
    }
    aml = {
        **base, 'event_id': 'F-AML', 'activity_entity_id': 'PLAYER:P-AML',
        'canonical_player_id': 'P-AML',
    }
    safer = {
        **base, 'event_id': 'F-SG', 'activity_entity_id': 'PLAYER:P-SG',
        'canonical_player_id': 'P-SG', 'unified_spending_acceleration': 3.0,
        'unified_net_loss_7d': 100.0, 'unified_net_loss_30d': 100.0,
        'unified_session_frequency_change': 3.0,
        'unified_wager_volume_7d': 100.0, 'unified_wager_volume_30d': 100.0,
        'peer_deposit_robust_z': 5.0,
    }
    missing = {
        key: np.nan for key in base
        if key not in {'timestamp', 'channel', 'data_sufficiency_status'}
    }
    missing.update({
        'timestamp': pd.Timestamp('2026-06-30'), 'channel': 'land_casino',
        'event_id': 'F-MISSING', 'activity_entity_id': 'SOURCE:land_casino:L-1',
        'canonical_player_id': None, 'data_sufficiency_status': 'limited',
    })
    return pd.DataFrame([aml, safer, missing])


def _model_rows():
    return pd.DataFrame([
        {'activity_entity_id': 'PLAYER:P-AML', 'timestamp': '2026-06-30',
         'anomaly_severity': 100.0, 'behavioral_change_score': 0.0},
        {'activity_entity_id': 'PLAYER:P-SG', 'timestamp': '2026-06-30',
         'anomaly_severity': 0.0, 'behavioral_change_score': 0.0},
        {'activity_entity_id': 'SOURCE:land_casino:L-1', 'timestamp': '2026-06-30',
         'anomaly_severity': np.nan, 'behavioral_change_score': np.nan},
    ])


def _transactions():
    rows = []
    for event_id, timestamp, deposit, withdrawal, wager, region in [
        ('T-BASE', '2026-05-01', 100, 0, 100, 'ON'),
        ('T-DEPOSIT', '2026-06-29', 1000, 0, 0, 'QC'),
        ('T-WITHDRAW', '2026-06-30', 0, 900, 10, 'QC'),
    ]:
        rows.append({
            'activity_entity_id': 'PLAYER:P-AML', 'event_id': event_id,
            'source_event_id': event_id, 'transaction_reference': event_id,
            'channel': 'online_casino', 'timestamp': pd.Timestamp(timestamp),
            'observed': True, 'deposit_amount': deposit,
            'withdrawal_amount': withdrawal, 'wager_amount': wager,
            'payout_amount': 0, 'amount': max(deposit, withdrawal, wager),
            'region': region,
        })
    return pd.DataFrame(rows)


def _network_nodes():
    return pd.DataFrame([{
        'node_id': 'online_casino:A-1', 'identity_status': 'confirmed',
        'canonical_player_id': 'P-AML', 'maximum_investigation_strength': .8,
    }])


@pytest.fixture()
def scored():
    return score_risk_indices(
        _feature_rows(), _model_rows(), transactions=_transactions(),
        network_nodes=_network_nodes(), review_capacity=2,
    )


def test_scores_are_bounded_and_contributions_reconcile(scored):
    assert scored.aml_fraud_score.between(0, 100).all()
    assert scored.safer_gambling_score.between(0, 100).all()
    for row in scored.itertuples():
        for category, score_column in [
            ('aml_fraud', 'aml_fraud_score'),
            ('safer_gambling', 'safer_gambling_score'),
        ]:
            ledger = row.signal_contributions[category]
            contributions = [
                value['score_contribution'] for value in ledger.values()
                if isinstance(value, dict) and value['score_contribution'] is not None
            ]
            assert sum(contributions) == pytest.approx(getattr(row, score_column))


def test_missing_data_is_explicit_and_not_folded_into_confidence(scored):
    row = scored.set_index('activity_entity_id').loc['SOURCE:land_casino:L-1']
    assert row.aml_fraud_score == 0
    assert row.safer_gambling_score == 0
    assert not row.signal_contributions['aml_fraud']['score_available']
    assert not row.signal_contributions['safer_gambling']['score_available']
    assert row.signal_contributions['aml_fraud']['missing_signals']
    assert row.data_coverage['data_sufficiency_status'] == 'limited'
    assert pd.isna(row.identity_confidence)


def test_risk_categories_are_independent_and_supporting_ids_are_preserved(scored):
    indexed = scored.set_index('activity_entity_id')
    aml = indexed.loc['PLAYER:P-AML']
    safer = indexed.loc['PLAYER:P-SG']
    assert aml.aml_fraud_score > aml.safer_gambling_score
    assert safer.safer_gambling_score > safer.aml_fraud_score
    assert {'T-DEPOSIT', 'T-WITHDRAW'}.issubset(aml.supporting_transaction_ids)


@pytest.mark.parametrize('column', ['ground_truth_label', 'scenario_type'])
def test_ground_truth_never_enters_scoring(column):
    contaminated = _feature_rows().assign(**{column: True})
    with pytest.raises(ValueError, match='Ground-truth columns'):
        score_risk_indices(contaminated, _model_rows())


@pytest.mark.parametrize('column', ['age_band', 'region'])
def test_protected_profile_attributes_are_rejected(column):
    contaminated = _feature_rows().assign(**{column: 'not-for-scoring'})
    with pytest.raises(ValueError, match='Protected demographic'):
        score_risk_indices(contaminated, _model_rows())


@pytest.mark.parametrize('capacity', [25, 50, 100])
def test_review_capacity_and_rank_are_deterministic(capacity):
    candidates = pd.DataFrame({
        'activity_entity_id': [f'PLAYER:P-{number:03}' for number in range(120)],
        'aml_fraud_score': np.arange(120, dtype=float) % 101,
        'safer_gambling_score': np.arange(119, -1, -1, dtype=float) % 101,
    })
    ranked = prioritize_investigations(candidates, review_capacity=capacity)
    assert ranked.priority_rank.tolist() == list(range(1, 121))
    assert ranked.selected_for_review.sum() == capacity


def test_identity_confidence_does_not_scale_risk_score():
    features = _feature_rows().iloc[[0]].copy()
    models = _model_rows().iloc[[0]].copy()
    high = score_risk_indices(features, models)
    features['identity_confidence'] = .2
    low = score_risk_indices(features, models)
    assert high.aml_fraud_score.iloc[0] == low.aml_fraud_score.iloc[0]
    assert high.identity_confidence.iloc[0] != low.identity_confidence.iloc[0]


def test_signal_strengths_are_capped_and_graded():
    result = score_risk_indices(_feature_rows(), _model_rows(), transactions=_transactions())
    values = []
    for ledger in result.signal_contributions:
        for category in ('aml_fraud', 'safer_gambling'):
            values.extend(
                item['signal_value'] for item in ledger[category].values()
                if isinstance(item, dict) and item['signal_value'] is not None
            )
    assert max(values) < 1.0
    assert len({round(value, 3) for value in values if value > 0}) > 1
