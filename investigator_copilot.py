"""Deterministic, evidence-grounded investigation cases for Sentinel X.

This module does not call an LLM and does not make account decisions. It turns
Phase 7 scoring evidence into a human-review case, and stores local prototype
workflow actions in a mutable mapping such as ``st.session_state``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, MutableMapping

import numpy as np
import pandas as pd

from signals import GROUND_TRUTH_COLUMNS


VALID_STATUSES = ('New', 'Under Review', 'Escalated', 'Closed')
VALID_ACTIONS = ('Start Review', 'Confirm', 'Dismiss', 'Add Note', 'Reopen')
STORE_KEY = 'sentinel_x_investigations'

SIGNAL_LABELS = {
    'transaction_anomaly': 'transaction-pattern anomaly',
    'unusual_fund_movement': 'rapid movement of funds with limited observed play',
    'network_relationships': 'multi-account relationship evidence',
    'geographic_anomaly': 'change in observed transaction location',
    'behavioral_change': "change from the player's own history",
    'cross_channel_inconsistency': 'cross-channel inconsistency',
    'deposit_escalation': 'deposit escalation',
    'observed_net_loss_increase': 'increase in reliably observed net losses',
    'session_frequency_increase': 'increase in session frequency',
    'gambling_intensity_change': 'change in gambling intensity',
    'historical_pattern_change': 'deviation from historical behavioral peers',
}


def escape_currency_markdown(text: str) -> str:
    """Prevent currency markers in generated prose from invoking Markdown math."""
    return str(text).replace('$', r'\$')


@dataclass(frozen=True)
class InvestigationCase:
    case_id: str
    player_id: str | None
    activity_entity_id: str
    status: str
    aml_fraud_score: float
    safer_gambling_score: float
    risk_band: str
    priority_rank: int | None
    why_flagged: tuple[str, ...]
    historical_changes: tuple[str, ...]
    contributing_channels: tuple[str, ...]
    supporting_transactions: tuple[dict, ...]
    evidence_for_concern: tuple[str, ...]
    evidence_against_or_alternatives: tuple[str, ...]
    missing_or_uncertain: tuple[str, ...]
    recommended_next_steps: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _reject_labels(frame_or_row, name: str):
    columns = set(frame_or_row.index) if isinstance(frame_or_row, pd.Series) else set(
        frame_or_row.columns
    )
    leaked = GROUND_TRUTH_COLUMNS.intersection(columns)
    if leaked:
        raise ValueError(
            f'Ground-truth columns cannot enter investigator cases ({name}): {sorted(leaked)}'
        )


def _number(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _value(item: dict | None) -> float:
    if not isinstance(item, dict) or not item.get('available', False):
        return np.nan
    return _number(item.get('signal_value'))


def _legacy_ledgers(row: pd.Series) -> dict[str, dict]:
    """Adapt the starter dashboard's transparent signals without inventing data."""
    def item(value):
        value = _number(value)
        return {
            'available': np.isfinite(value),
            'signal_value': None if not np.isfinite(value) else value,
            'score_contribution': None,
        }

    aml = {
        'transaction_anomaly': item(np.nan),
        'unusual_fund_movement': item(np.nan),
        'network_relationships': item(row.get('network_signal')),
        'geographic_anomaly': item(row.get('geo_signal')),
        'behavioral_change': item(row.get('change_signal')),
        'cross_channel_inconsistency': item(row.get('cross_signal')),
    }
    safer = {
        'deposit_escalation': item(row.get('change_signal')),
        'observed_net_loss_increase': item(row.get('loss_signal')),
        'session_frequency_increase': item(np.nan),
        'gambling_intensity_change': item(np.nan),
        'historical_pattern_change': item(np.nan),
    }
    for ledger in (aml, safer):
        ledger['missing_signals'] = [
            name for name, value in ledger.items()
            if isinstance(value, dict) and not value['available']
        ]
    return {'aml_fraud': aml, 'safer_gambling': safer}


def _ledgers(row: pd.Series) -> dict[str, dict]:
    value = row.get('signal_contributions')
    if isinstance(value, dict) and {'aml_fraud', 'safer_gambling'}.issubset(value):
        return value
    return _legacy_ledgers(row)


def _signal_evidence(ledgers: dict[str, dict], threshold: float = .30) -> list[str]:
    evidence = []
    for category, label in (
        ('aml_fraud', 'AML/Fraud'), ('safer_gambling', 'Safer Gambling')
    ):
        for signal, item in ledgers[category].items():
            if signal not in SIGNAL_LABELS:
                continue
            value = _value(item)
            if np.isfinite(value) and value >= threshold:
                contribution = _number(item.get('score_contribution'))
                points = (
                    f' and contributed {contribution:.2f} points'
                    if np.isfinite(contribution) else ''
                )
                evidence.append(
                    f'{label} evidence: '
                    f'{SIGNAL_LABELS[signal]} measured {value:.0%}{points}.'
                )
    return evidence


def _observed_mitigation(
    ledgers: dict[str, dict],
    context: pd.Series,
    entity_transactions: pd.DataFrame,
) -> list[str]:
    """Return specific observed context, never generic low-signal filler."""
    output = []
    tenure = _number(context.get('account_tenure_days'))
    if np.isfinite(tenure) and tenure >= 365:
        output.append(
            f'Observed mitigating evidence: the account has {int(tenure):,} days of recorded tenure.'
        )
    stability = _number(context.get('unified_behavioral_stability'))
    if np.isfinite(stability) and stability >= .70:
        output.append(
            f'Observed mitigating evidence: historical spending stability measured {stability:.0%}.'
        )
    percentile = _number(context.get('peer_deposit_percentile'))
    if np.isfinite(percentile) and percentile <= .75:
        output.append(
            f'Observed mitigating evidence: 30-day deposits were at the {percentile:.0%} '
            'percentile of the behavioral peer group.'
        )
    network = _value(ledgers['aml_fraud'].get('network_relationships'))
    if np.isfinite(network) and network <= .05:
        output.append(
            'Observed mitigating evidence: no shared device or payment-instrument '
            'infrastructure was found in the supplied relationship graph.'
        )
    if not entity_transactions.empty and 'timestamp' in entity_transactions:
        timestamps = pd.to_datetime(entity_transactions.timestamp, errors='coerce').dropna()
        holidays = {
            pd.Timestamp('2026-01-01').date(), pd.Timestamp('2026-02-16').date(),
            pd.Timestamp('2026-04-03').date(), pd.Timestamp('2026-05-18').date(),
            pd.Timestamp('2026-07-01').date(), pd.Timestamp('2026-08-03').date(),
            pd.Timestamp('2026-09-07').date(), pd.Timestamp('2026-10-12').date(),
            pd.Timestamp('2026-12-25').date(), pd.Timestamp('2026-12-28').date(),
        }
        overlap = timestamps.dt.dayofweek.ge(5) | timestamps.dt.date.map(lambda value: value in holidays)
        if len(overlap) and overlap.any():
            output.append(
                f'Observed mitigating context: {int(overlap.sum())} of {len(overlap)} supplied '
                'events occurred on a weekend or Ontario public-holiday date.'
            )
    return output


def _hypotheses(ledgers: dict[str, dict]) -> list[str]:
    active = {
        signal
        for category in ('aml_fraud', 'safer_gambling')
        for signal, item in ledgers[category].items()
        if signal in SIGNAL_LABELS and np.isfinite(_value(item)) and _value(item) >= .30
    }
    hypotheses = []
    if active & {'unusual_fund_movement', 'transaction_anomaly'}:
        hypotheses.append(
            'Unverified hypothesis: the fund pattern could reflect ordinary wallet management '
            'or withdrawal of winnings; source records and funding ownership must be checked.'
        )
    if 'network_relationships' in active:
        hypotheses.append(
            'Unverified hypothesis: shared identifiers could reflect a household or shared device; '
            'the relationship must be verified and is not proof of fraud.'
        )
    if 'geographic_anomaly' in active:
        hypotheses.append(
            'Unverified hypothesis: travel or location-data limitations could explain the observed '
            'location change; neither explanation is established by the available records.'
        )
    if active & {
        'deposit_escalation', 'observed_net_loss_increase',
        'session_frequency_increase', 'gambling_intensity_change',
    }:
        hypotheses.append(
            'Unverified hypothesis: a temporary change in discretionary activity could contribute '
            'to the increase; affordability and player context are not established here.'
        )
    return hypotheses


def _entity_transactions(row: pd.Series, transactions: pd.DataFrame) -> pd.DataFrame:
    entity = str(row.get('activity_entity_id', ''))
    player = row.get('player_id')
    canonical = row.get('canonical_player_id')
    if 'activity_entity_id' in transactions and entity:
        return transactions[transactions.activity_entity_id.astype(str).eq(entity)].copy()
    identifier = canonical if pd.notna(canonical) else player
    if 'player_id' in transactions and pd.notna(identifier):
        return transactions[transactions.player_id.eq(identifier)].copy()
    return transactions.iloc[0:0].copy()


def _event_id_column(transactions: pd.DataFrame) -> str | None:
    for column in ('event_id', 'tx_id', 'source_event_id'):
        if column in transactions:
            return column
    return None


def _event_amount(record: pd.Series) -> float | None:
    direct = _number(record.get('amount'))
    if np.isfinite(direct):
        return direct
    values = [
        abs(_number(record.get(name), 0.0))
        for name in ('deposit_amount', 'withdrawal_amount', 'wager_amount',
                     'payout_amount', 'deposit', 'wager', 'net_loss')
    ]
    return max(values) if values and max(values) > 0 else None


def _event_type(record: pd.Series) -> str:
    if pd.notna(record.get('event_type')):
        return str(record.event_type)
    for field, label in (
        ('deposit', 'deposit'), ('wager', 'wager'), ('net_loss', 'observed net loss')
    ):
        if _number(record.get(field), 0) > 0:
            return label
    return 'observed activity'


def _supporting_records(row: pd.Series, transactions: pd.DataFrame, limit: int = 8):
    entity_transactions = _entity_transactions(row, transactions)
    if 'observed' in entity_transactions:
        entity_transactions = entity_transactions[entity_transactions.observed.fillna(False)]
    id_column = _event_id_column(entity_transactions)
    requested = row.get('supporting_transaction_ids', [])
    requested = list(requested) if isinstance(requested, (list, tuple, set, np.ndarray)) else []
    if id_column and requested:
        selected = entity_transactions[
            entity_transactions[id_column].astype(str).isin(map(str, requested))
        ].copy()
        missing_ids = sorted(set(map(str, requested)).difference(selected[id_column].astype(str)))
    else:
        selected = entity_transactions.copy()
        missing_ids = []
    if selected.empty or id_column is None:
        return [], missing_ids
    selected['_copilot_amount'] = selected.apply(_event_amount, axis=1).fillna(0)
    ordering = ['_copilot_amount']
    ascending = [False]
    if 'timestamp' in selected:
        ordering.append('timestamp')
        ascending.append(False)
    selected = selected.sort_values(ordering, ascending=ascending).head(limit)
    output = []
    for _, record in selected.iterrows():
        timestamp = pd.to_datetime(record.get('timestamp'), errors='coerce')
        output.append({
            'transaction_id': str(record[id_column]),
            'timestamp': None if pd.isna(timestamp) else timestamp.isoformat(),
            'channel': str(record.get('channel', 'unknown')),
            'event_type': _event_type(record),
            'amount': _event_amount(record),
        })
    return output, missing_ids


def _latest_context(row: pd.Series, feature_history: pd.DataFrame | None) -> pd.Series:
    if feature_history is None or feature_history.empty:
        return row
    _reject_labels(feature_history, 'feature_history')
    entity = str(row.get('activity_entity_id', ''))
    if 'activity_entity_id' in feature_history and entity:
        history = feature_history[feature_history.activity_entity_id.astype(str).eq(entity)]
    elif 'player_id' in feature_history:
        history = feature_history[feature_history.player_id.eq(row.get('player_id'))]
    else:
        history = feature_history.iloc[0:0]
    if history.empty:
        return row
    ordering = ['timestamp'] + (['event_id'] if 'event_id' in history else [])
    latest = history.sort_values(ordering).iloc[-1]
    return pd.concat([row, latest[~latest.index.isin(row.index)]])


def _historical_changes(context: pd.Series) -> list[str]:
    output = []
    acceleration = _number(context.get('unified_spending_acceleration'))
    if np.isfinite(acceleration):
        output.append(
            f'Observed 7-day deposit rate changed by {acceleration:+.0%} versus the preceding '
            '23 days.'
        )
    elif np.isfinite(_number(context.get('change_ratio'))):
        output.append(
            f'Recent observed weekly deposits were {_number(context.change_ratio):.2f} times '
            'the historical weekly baseline.'
        )
    sessions = _number(context.get('unified_session_frequency_change'))
    if np.isfinite(sessions):
        output.append(
            f'Observed 7-day session rate changed by {sessions:+.0%} versus the preceding 23 days.'
        )
    loss7, loss30 = (
        _number(context.get('unified_net_loss_7d')),
        _number(context.get('unified_net_loss_30d')),
    )
    if np.isfinite(loss7) and np.isfinite(loss30):
        output.append(
            f'Reliably observed net loss was ${loss7:,.2f} in 7 days and ${loss30:,.2f} '
            'across 30 days.'
        )
    return output


def _missing_evidence(row: pd.Series, ledgers: dict, missing_ids: list[str]) -> list[str]:
    missing = []
    for category in ('aml_fraud', 'safer_gambling'):
        names = ledgers[category].get('missing_signals', [])
        if names:
            labels = ', '.join(SIGNAL_LABELS.get(name, name) for name in names)
            missing.append(f'{category.replace("_", " ").title()} signals unavailable: {labels}.')
    coverage = row.get('data_coverage')
    if isinstance(coverage, dict):
        channel = _number(coverage.get('channel_coverage_30d'))
        if np.isfinite(channel) and channel < 1:
            missing.append(f'Observed 30-day channel coverage is {channel:.0%}, not complete coverage.')
        unobserved = _number(coverage.get('known_unobserved_records_30d'))
        if np.isfinite(unobserved) and unobserved > 0:
            missing.append(f'{int(unobserved)} source records are explicitly marked unobserved.')
        if coverage.get('data_sufficiency_status') in {'limited', 'unknown'}:
            missing.append('Historical context is limited; missing activity is not zero activity.')
    elif row.get('data_quality') == 'limited':
        missing.append('Historical context is limited; missing activity is not zero activity.')
    identity = _number(row.get('identity_confidence'))
    if not np.isfinite(identity):
        missing.append('Identity confidence is unavailable.')
    elif identity < .90:
        missing.append(
            f'Identity confidence is {identity:.0%}; uncertain records were not merged automatically.'
        )
    if missing_ids:
        missing.append(
            'Supporting IDs not found in the supplied transaction frame: ' + ', '.join(missing_ids) + '.'
        )
    return list(dict.fromkeys(missing))


def _recommendations(row: pd.Series, ledgers: dict, supporting: list[dict]) -> list[str]:
    steps = []
    if supporting:
        steps.append('Verify the cited transactions against their source records and ledger sequence.')
    identity = _number(row.get('identity_confidence'))
    if not np.isfinite(identity) or identity < .90:
        steps.append('Resolve identity uncertainty before combining additional account histories.')
    if _value(ledgers['aml_fraud'].get('network_relationships')) >= .30:
        steps.append('Review the typed relationship evidence; do not infer fraud from a shared identifier alone.')
    if _number(row.get('safer_gambling_score'), 0) >= 35:
        steps.append('Have an authorized safer-gambling reviewer assess the observed change and available context.')
    if _number(row.get('aml_fraud_score'), 0) >= 35:
        steps.append('Have an authorized AML/fraud investigator assess source-of-funds and transaction context.')
    steps.append(
        "Record the investigator's reasoning and disposition; take no automated account, restriction, "
        'or player-contact action from this recommendation.'
    )
    return steps


def build_investigation_case(
    scored_row: pd.Series | dict,
    transactions: pd.DataFrame,
    *,
    feature_history: pd.DataFrame | None = None,
    status: str = 'New',
) -> InvestigationCase:
    """Build one deterministic case using only supplied operational evidence."""
    row = scored_row if isinstance(scored_row, pd.Series) else pd.Series(scored_row)
    _reject_labels(row, 'scored_row')
    _reject_labels(transactions, 'transactions')
    if status not in VALID_STATUSES:
        raise ValueError(f'Invalid investigation status: {status}')
    entity_value = row.get('activity_entity_id')
    if pd.isna(entity_value) or not str(entity_value).strip():
        entity_value = f"PLAYER:{row.get('player_id', 'UNKNOWN')}"
    entity = str(entity_value)
    player = row.get('player_id')
    if pd.isna(player):
        player = row.get('canonical_player_id')
    player = None if pd.isna(player) else str(player)
    ledgers = _ledgers(row)
    context = _latest_context(row, feature_history)
    supporting, missing_ids = _supporting_records(row, transactions)
    channels = tuple(sorted({item['channel'] for item in supporting if item['channel'] != 'unknown'}))
    if not channels:
        entity_events = _entity_transactions(row, transactions)
        if 'channel' in entity_events:
            channels = tuple(sorted(entity_events.channel.dropna().astype(str).unique()))

    aml = _number(row.get('aml_fraud_score'), 0.0)
    safer = _number(row.get('safer_gambling_score'), 0.0)
    rank = _number(row.get('priority_rank'))
    why = [
        f'The case was prioritized with AML/Fraud index {aml:.2f} and '
        f'Safer Gambling index {safer:.2f}; these are review indices, not probabilities.'
    ]
    if np.isfinite(rank):
        why.append(f'It ranked {int(rank)} in the configured investigation queue.')
    evidence = _signal_evidence(ledgers)
    if not evidence:
        evidence = ['No individual supplied signal exceeded the narrative concern threshold.']
    entity_transactions = _entity_transactions(row, transactions)
    alternatives = _observed_mitigation(ledgers, context, entity_transactions) + _hypotheses(ledgers)
    if not alternatives:
        alternatives = [
            'Unverified hypothesis: additional legitimate context may exist, but none is established '
            'by the supplied records.'
        ]
    missing = _missing_evidence(row, ledgers, missing_ids)
    return InvestigationCase(
        case_id=f'CASE-{entity}', player_id=player, activity_entity_id=entity,
        status=status, aml_fraud_score=aml, safer_gambling_score=safer,
        risk_band=str(row.get('risk_band', 'unassigned')),
        priority_rank=int(rank) if np.isfinite(rank) else None,
        why_flagged=tuple(why), historical_changes=tuple(_historical_changes(context)),
        contributing_channels=channels, supporting_transactions=tuple(supporting),
        evidence_for_concern=tuple(evidence),
        evidence_against_or_alternatives=tuple(dict.fromkeys(alternatives)),
        missing_or_uncertain=tuple(missing),
        recommended_next_steps=tuple(_recommendations(row, ledgers, supporting)),
    )


def build_flagged_cases(
    scored: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    feature_history: pd.DataFrame | None = None,
) -> list[InvestigationCase]:
    """Build cases for the capacity-selected queue, or every row if no flag exists."""
    _reject_labels(scored, 'scored')
    flagged = scored[scored.selected_for_review.fillna(False)] \
        if 'selected_for_review' in scored else scored
    return [
        build_investigation_case(row, transactions, feature_history=feature_history)
        for _, row in flagged.iterrows()
    ]


class SessionInvestigationStore:
    """Local workflow store backed by ``dict`` or Streamlit session state."""

    def __init__(
        self,
        state: MutableMapping,
        clock: Callable[[], datetime] | None = None,
    ):
        self.state = state
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if STORE_KEY not in self.state:
            self.state[STORE_KEY] = {}

    @property
    def cases(self) -> dict:
        return self.state[STORE_KEY]

    def initialize(self, case: InvestigationCase) -> dict:
        if case.case_id not in self.cases:
            self.cases[case.case_id] = {
                'status': case.status,
                'notes': [],
                'audit_history': [{
                    'timestamp': self.clock().isoformat(),
                    'investigator': 'system',
                    'action': 'Case Created',
                    'from_status': None,
                    'to_status': case.status,
                    'note': '',
                }],
            }
        return self.cases[case.case_id]

    def record_action(
        self,
        case: InvestigationCase,
        action: str,
        *,
        investigator: str = 'Local Investigator',
        note: str = '',
    ) -> dict:
        if action not in VALID_ACTIONS:
            raise ValueError(f'Invalid investigation action: {action}')
        record = self.initialize(case)
        old_status = record['status']
        transitions = {
            'Start Review': 'Under Review',
            'Confirm': 'Escalated',
            'Dismiss': 'Closed',
            'Add Note': old_status,
            'Reopen': 'Under Review',
        }
        new_status = transitions[action]
        if action == 'Reopen' and old_status != 'Closed':
            raise ValueError('Only a closed investigation can be reopened')
        if note.strip():
            record['notes'].append({
                'timestamp': self.clock().isoformat(),
                'investigator': investigator,
                'text': note.strip(),
            })
        record['status'] = new_status
        record['audit_history'].append({
            'timestamp': self.clock().isoformat(),
            'investigator': investigator,
            'action': action,
            'from_status': old_status,
            'to_status': new_status,
            'note': note.strip(),
        })
        return record


def render_investigator_controls(case: InvestigationCase, session_state=None):
    """Render local-only Streamlit controls and an immutable-style audit trail."""
    import streamlit as st

    state = st.session_state if session_state is None else session_state
    store = SessionInvestigationStore(state)
    record = store.initialize(case)
    st.markdown(f"**Investigation status:** {record['status']}")
    investigator = st.text_input('Investigator', value='Local Investigator', key=f'{case.case_id}-user')
    note = st.text_area('Investigator notes', key=f'{case.case_id}-note')
    first, second, third = st.columns(3)
    if first.button('Start / Continue Review', key=f'{case.case_id}-review'):
        store.record_action(case, 'Start Review', investigator=investigator, note=note)
        st.rerun()
    if second.button('Confirm for Escalation', key=f'{case.case_id}-confirm'):
        store.record_action(case, 'Confirm', investigator=investigator, note=note)
        st.rerun()
    if third.button('Dismiss Review', key=f'{case.case_id}-dismiss'):
        store.record_action(case, 'Dismiss', investigator=investigator, note=note)
        st.rerun()
    if st.button('Save Note', key=f'{case.case_id}-save-note'):
        store.record_action(case, 'Add Note', investigator=investigator, note=note)
        st.rerun()
    st.caption(
        'These controls record a local human-review disposition only. They do not freeze an account, '
        'apply gambling restrictions, or contact a player.'
    )
    if record['notes']:
        st.markdown('#### Investigator notes')
        st.dataframe(pd.DataFrame(record['notes']), width='stretch', hide_index=True)
    st.markdown('#### Audit history')
    st.dataframe(pd.DataFrame(record['audit_history']), width='stretch', hide_index=True)

