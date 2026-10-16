from datetime import datetime, timezone

import pandas as pd
import pytest

from investigator_copilot import (
    SessionInvestigationStore,
    build_flagged_cases,
    build_investigation_case,
    escape_currency_markdown,
)


def _signal(value, points):
    return {
        'available': value is not None,
        'signal_value': value,
        'configured_weight': .2,
        'normalized_weight': .2 if value is not None else None,
        'score_contribution': points if value is not None else None,
    }


def _scored_row():
    aml = {
        'transaction_anomaly': _signal(.80, 20.0),
        'unusual_fund_movement': _signal(.70, 14.0),
        'network_relationships': _signal(.60, 12.0),
        'geographic_anomaly': _signal(.50, 5.0),
        'behavioral_change': _signal(.40, 6.0),
        'cross_channel_inconsistency': _signal(.10, 1.0),
        'missing_signals': [],
    }
    safer = {
        'deposit_escalation': _signal(.90, 22.5),
        'observed_net_loss_increase': _signal(.70, 17.5),
        'session_frequency_increase': _signal(None, None),
        'gambling_intensity_change': _signal(.10, 2.0),
        'historical_pattern_change': _signal(.40, 6.0),
        'missing_signals': ['session_frequency_increase'],
    }
    return pd.Series({
        'player_id': 'P-1', 'canonical_player_id': 'P-1',
        'activity_entity_id': 'PLAYER:P-1', 'aml_fraud_score': 58.0,
        'safer_gambling_score': 48.0, 'priority_rank': 3, 'risk_band': 'moderate',
        'selected_for_review': True,
        'signal_contributions': {'aml_fraud': aml, 'safer_gambling': safer},
        'data_coverage': {
            'channel_coverage_30d': 2 / 3,
            'observed_record_fraction_30d': .9,
            'missing_field_fraction_30d': .1,
            'known_unobserved_records_30d': 2,
            'data_sufficiency_status': 'moderate',
        },
        'identity_confidence': .85,
        'supporting_transaction_ids': ['T-1', 'T-2', 'NOT-SUPPLIED'],
    })


def _transactions():
    return pd.DataFrame([
        {
            'event_id': 'T-1', 'activity_entity_id': 'PLAYER:P-1',
            'timestamp': pd.Timestamp('2026-06-29 10:00'), 'channel': 'online_casino',
            'event_type': 'deposit', 'amount': 1000.0, 'observed': True,
        },
        {
            'event_id': 'T-2', 'activity_entity_id': 'PLAYER:P-1',
            'timestamp': pd.Timestamp('2026-06-30 11:00'), 'channel': 'sportsbook',
            'event_type': 'withdrawal', 'amount': 850.0, 'observed': True,
        },
        {
            'event_id': 'T-OTHER', 'activity_entity_id': 'PLAYER:P-2',
            'timestamp': pd.Timestamp('2026-06-30 12:00'), 'channel': 'land_casino',
            'event_type': 'buy_in', 'amount': 5000.0, 'observed': True,
        },
    ])


def _features():
    return pd.DataFrame([{
        'event_id': 'F-1', 'activity_entity_id': 'PLAYER:P-1',
        'timestamp': pd.Timestamp('2026-06-30'),
        'unified_spending_acceleration': 2.0,
        'unified_session_frequency_change': .5,
        'unified_net_loss_7d': 350.0,
        'unified_net_loss_30d': 600.0,
    }])


def test_case_narrative_is_grounded_in_actual_scores_features_and_transactions():
    case = build_investigation_case(
        _scored_row(), _transactions(), feature_history=_features()
    )

    assert 'AML/Fraud index 58.00' in case.why_flagged[0]
    assert any('transaction-pattern anomaly measured 80%' in text for text in case.evidence_for_concern)
    assert any('contributed 20.00 points' in text for text in case.evidence_for_concern)
    assert any('+200%' in text for text in case.historical_changes)
    assert any('$350.00' in text and '$600.00' in text for text in case.historical_changes)
    assert case.contributing_channels == ('online_casino', 'sportsbook')
    assert {item['transaction_id'] for item in case.supporting_transactions} == {'T-1', 'T-2'}
    assert all(item['transaction_id'] != 'T-OTHER' for item in case.supporting_transactions)
    assert case.recommended_next_steps


def test_missing_support_and_identity_uncertainty_are_disclosed_not_fabricated():
    case = build_investigation_case(_scored_row(), _transactions())

    uncertainty = ' '.join(case.missing_or_uncertain)
    assert 'NOT-SUPPLIED' in uncertainty
    assert '85%' in uncertainty
    assert 'not complete coverage' in uncertainty
    assert '2 source records' in uncertainty
    assert all('NOT-SUPPLIED' != item['transaction_id'] for item in case.supporting_transactions)


def test_alternative_explanations_are_explicitly_unverified():
    case = build_investigation_case(_scored_row(), _transactions())
    hypotheses = [
        text for text in case.evidence_against_or_alternatives
        if not text.startswith('Observed mitigating evidence:')
    ]

    assert hypotheses
    assert all(text.startswith('Unverified hypothesis:') for text in hypotheses)
    assert any('not proof of fraud' in text for text in hypotheses)
    assert all('automatically freeze' not in text.lower() for text in case.recommended_next_steps)
    assert any('take no automated account' in text for text in case.recommended_next_steps)


def test_labels_are_rejected_at_case_boundary():
    contaminated = _scored_row().copy()
    contaminated['ground_truth_label'] = True
    with pytest.raises(ValueError, match='Ground-truth columns'):
        build_investigation_case(contaminated, _transactions())


def test_flagged_case_builder_obeys_review_selection():
    selected = _scored_row()
    unselected = _scored_row().copy()
    unselected['player_id'] = 'P-2'
    unselected['activity_entity_id'] = 'PLAYER:P-2'
    unselected['selected_for_review'] = False
    cases = build_flagged_cases(pd.DataFrame([selected, unselected]), _transactions())

    assert [case.player_id for case in cases] == ['P-1']


def test_local_workflow_tracks_status_notes_and_audit_history():
    case = build_investigation_case(_scored_row(), _transactions())
    state = {}
    fixed = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
    store = SessionInvestigationStore(state, clock=lambda: fixed)

    record = store.initialize(case)
    assert record['status'] == 'New'
    store.record_action(case, 'Start Review', investigator='Analyst A', note='Checking source records')
    assert record['status'] == 'Under Review'
    assert record['notes'][0]['text'] == 'Checking source records'
    store.record_action(case, 'Confirm', investigator='Analyst A', note='Manual escalation')
    assert record['status'] == 'Escalated'
    store.record_action(case, 'Dismiss', investigator='Analyst B', note='Resolved after review')
    assert record['status'] == 'Closed'
    store.record_action(case, 'Reopen', investigator='Analyst B')
    assert record['status'] == 'Under Review'
    assert [item['action'] for item in record['audit_history']] == [
        'Case Created', 'Start Review', 'Confirm', 'Dismiss', 'Reopen'
    ]
    assert all(item['timestamp'] == fixed.isoformat() for item in record['audit_history'])


def test_invalid_workflow_actions_are_rejected():
    case = build_investigation_case(_scored_row(), _transactions())
    store = SessionInvestigationStore({})
    with pytest.raises(ValueError, match='Invalid investigation action'):
        store.record_action(case, 'Freeze Account')
    with pytest.raises(ValueError, match='Only a closed'):
        store.record_action(case, 'Reopen')


def test_currency_is_escaped_only_for_markdown_rendering():
    assert escape_currency_markdown('Observed $350.00 and $600.00') == (
        r'Observed \$350.00 and \$600.00'
    )

