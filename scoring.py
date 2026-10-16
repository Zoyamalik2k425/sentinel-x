"""Explainable risk indices and capacity-aware investigation prioritization.

The Phase 7 API is :func:`score_risk_indices`. The older :func:`score` and
:func:`explain` functions remain available for the starter Streamlit app.
Synthetic scenario labels and protected profile attributes are rejected as
inputs to operational scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from configuration import DEFAULT_MODEL_CONFIG, ModelConfig
from models import predict
from signals import GROUND_TRUTH_COLUMNS


AML_SIGNAL_NAMES = (
    'transaction_anomaly', 'unusual_fund_movement', 'network_relationships',
    'geographic_anomaly', 'behavioral_change', 'cross_channel_inconsistency',
)
SAFER_GAMBLING_SIGNAL_NAMES = (
    'deposit_escalation', 'observed_net_loss_increase',
    'session_frequency_increase', 'gambling_intensity_change',
    'historical_pattern_change',
)
PROTECTED_SCORING_COLUMNS = frozenset({
    'age', 'age_band', 'region', 'home_region', 'postal_code', 'gender',
    'income_band', 'occupation_category',
})


def _default_aml_weights() -> dict[str, float]:
    return {
        'transaction_anomaly': .40,
        'unusual_fund_movement': .30,
        'network_relationships': .20,
        'geographic_anomaly': .04,
        'behavioral_change': .03,
        'cross_channel_inconsistency': .03,
    }


def _default_safer_gambling_weights() -> dict[str, float]:
    return {
        'deposit_escalation': .25,
        'observed_net_loss_increase': .25,
        'session_frequency_increase': .15,
        'gambling_intensity_change': .20,
        'historical_pattern_change': .15,
    }


@dataclass(frozen=True)
class RiskScoringConfig:
    """Weights, risk bands, and analyst capacity for Phase 7 scoring.

    Weights are normalized over the configured signal set. Missing inputs
    never silently become observed zero-risk values: their contributions are
    ``None`` and the ledger reports evidence-weight coverage. If an entire
    index is unavailable its queue-compatible value is 0 and its ledger's
    ``score_available`` flag is false.
    """

    aml_weights: Mapping[str, float] = field(default_factory=_default_aml_weights)
    safer_gambling_weights: Mapping[str, float] = field(
        default_factory=_default_safer_gambling_weights
    )
    review_capacity: int = 50
    moderate_cutoff: float = 35.0
    high_cutoff: float = 60.0
    critical_cutoff: float = 80.0
    maximum_supporting_transactions: int = 8

    def __post_init__(self):
        _validate_weights('AML/Fraud', self.aml_weights, AML_SIGNAL_NAMES)
        _validate_weights(
            'Safer Gambling', self.safer_gambling_weights, SAFER_GAMBLING_SIGNAL_NAMES
        )
        if self.review_capacity < 1:
            raise ValueError('review_capacity must be positive')
        if not 0 <= self.moderate_cutoff < self.high_cutoff < self.critical_cutoff <= 100:
            raise ValueError('risk-band cutoffs must increase within 0-100')
        if self.maximum_supporting_transactions < 1:
            raise ValueError('maximum_supporting_transactions must be positive')


def _validate_weights(label: str, weights: Mapping[str, float], expected: tuple[str, ...]):
    if set(weights) != set(expected):
        raise ValueError(f'{label} weights must contain exactly {list(expected)}')
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError(f'{label} weights must be finite and non-negative')
    if sum(weights.values()) <= 0:
        raise ValueError(f'{label} weights must have a positive total')


DEFAULT_RISK_SCORING_CONFIG = RiskScoringConfig()


def _reject_forbidden_columns(frame: pd.DataFrame | None, name: str, *, profiles=False):
    if frame is None:
        return
    leaked = GROUND_TRUTH_COLUMNS.intersection(frame.columns)
    if leaked:
        raise ValueError(
            f'Ground-truth columns cannot enter risk scoring ({name}): {sorted(leaked)}'
        )
    # Transaction region measures change from the account's own observed
    # history. Registered/home geography is never consulted.
    if profiles:
        protected = PROTECTED_SCORING_COLUMNS.intersection(frame.columns)
        if protected:
            raise ValueError(
                f'Protected demographic columns cannot enter risk scoring ({name}): '
                f'{sorted(protected)}'
            )


def _number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _unit(value) -> float:
    value = _number(value)
    return np.nan if np.isnan(value) else float(np.clip(value, 0.0, 1.0))


def _latest_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {'activity_entity_id', 'timestamp'}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'Missing scoring feature columns: {sorted(missing)}')
    latest = features.copy()
    latest['timestamp'] = pd.to_datetime(latest.timestamp)
    ordering = ['timestamp'] + (['event_id'] if 'event_id' in latest.columns else [])
    return latest.sort_values(ordering).groupby('activity_entity_id', dropna=False).tail(1)


def _deduplicate_transactions(events: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated source rows and mirrored records before aggregation."""
    frame = events.copy()
    if {'channel', 'source_event_id'}.issubset(frame.columns):
        frame = frame.drop_duplicates(['channel', 'source_event_id'])
    elif 'event_id' in frame.columns:
        frame = frame.drop_duplicates('event_id')
    if 'transaction_reference' in frame.columns:
        def positive(column):
            if column not in frame:
                return pd.Series(False, index=frame.index)
            return pd.to_numeric(frame[column], errors='coerce').fillna(0).gt(0)
        direction = np.select(
            [positive('deposit_amount'), positive('withdrawal_amount'),
             positive('wager_amount'), positive('payout_amount')],
            ['deposit', 'withdrawal', 'wager', 'payout'], default='other',
        )
        frame = frame.assign(_direction=direction)
        duplicate_reference = frame.transaction_reference.notna() & frame.duplicated(
            ['transaction_reference', '_direction']
        )
        frame = frame.loc[~duplicate_reference].drop(columns='_direction')
    return frame


def _transaction_context(
    events: pd.DataFrame | None,
    maximum_ids: int,
) -> dict[str, dict[str, object]]:
    if events is None or events.empty:
        return {}
    required = {'activity_entity_id', 'event_id', 'timestamp'}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f'Missing scoring transaction columns: {sorted(missing)}')
    frame = _deduplicate_transactions(events)
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    if 'observed' in frame.columns:
        frame = frame[frame.observed.fillna(False)]
    if frame.empty:
        return {}
    entity = frame.activity_entity_id.astype(str)
    day_ns = 86_400_000_000_000
    timestamp_ns = frame.timestamp.astype('int64')
    as_of_ns = timestamp_ns.groupby(entity).transform('max')
    age_ns = as_of_ns - timestamp_ns
    recent = frame[age_ns.le(30 * day_ns)].copy()
    recent_entity = recent.activity_entity_id.astype(str)

    context = pd.DataFrame(index=pd.Index(entity.unique(), name='activity_entity_id'))
    totals = {}
    for column in ('deposit_amount', 'withdrawal_amount', 'wager_amount'):
        if column in recent:
            totals[column] = pd.to_numeric(recent[column], errors='coerce').groupby(
                recent_entity
            ).sum(min_count=1).reindex(context.index)
        else:
            totals[column] = pd.Series(np.nan, index=context.index)
    deposits, withdrawals, wagers = (
        totals['deposit_amount'], totals['withdrawal_amount'], totals['wager_amount']
    )
    withdrawal_ratio = (withdrawals / deposits).clip(0, 1)
    play_ratio = (wagers.fillna(0) / deposits).clip(0, 1)
    deposit_events = recent.assign(
        _is_deposit=pd.to_numeric(recent.get('deposit_amount'), errors='coerce').fillna(0).gt(0)
    ).groupby(recent_entity)._is_deposit.sum().reindex(context.index).fillna(0)
    repeated_deposit_signal = ((deposit_events - 3) / 9).clip(0, 1)
    limited_play_movement = withdrawal_ratio * (1 - play_ratio)
    context['unusual_fund_movement'] = pd.concat(
        [limited_play_movement, repeated_deposit_signal * (1 - play_ratio)], axis=1
    ).max(axis=1).where(deposits.gt(0) & withdrawals.notna())

    context['geographic_anomaly'] = np.nan
    if 'region' in frame:
        very_recent = frame[age_ns.le(7 * day_ns) & frame.region.notna()].copy()
        prior = frame[age_ns.gt(7 * day_ns) & frame.region.notna()].copy()
        if len(very_recent) and len(prior):
            baseline = prior.groupby(prior.activity_entity_id.astype(str)).region.agg(
                lambda values: values.astype(str).mode().iloc[0]
            )
            current_entity = very_recent.activity_entity_id.astype(str)
            mismatch = very_recent.region.astype(str).ne(current_entity.map(baseline))
            valid = current_entity.isin(baseline.index)
            geographic = mismatch[valid].groupby(current_entity[valid]).mean()
            context.loc[geographic.index, 'geographic_anomaly'] = geographic

    amount = (
        pd.to_numeric(recent['amount'], errors='coerce')
        if 'amount' in recent else pd.Series(np.nan, index=recent.index)
    )
    support_frame = recent.assign(
        _entity=recent_entity, _support_amount=amount.fillna(0)
    ).sort_values(
        ['_entity', '_support_amount', 'timestamp'], ascending=[True, False, False]
    )
    support = support_frame.drop_duplicates(['_entity', 'event_id']).groupby('_entity').head(
        maximum_ids
    ).groupby('_entity').event_id.agg(lambda values: values.astype(str).tolist())
    context['supporting_transaction_ids'] = support.reindex(context.index).map(
        lambda value: value if isinstance(value, list) else []
    )
    return {
        str(index): {
            'unusual_fund_movement': row.unusual_fund_movement,
            'geographic_anomaly': row.geographic_anomaly,
            'supporting_transaction_ids': row.supporting_transaction_ids,
        }
        for index, row in context.iterrows()
    }


def _network_context(nodes: pd.DataFrame | None) -> dict[str, float]:
    if nodes is None or nodes.empty:
        return {}
    required = {'node_id', 'identity_status', 'canonical_player_id'}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f'Missing network node columns: {sorted(missing)}')
    if 'maximum_investigation_strength' not in nodes.columns:
        # Old exports cannot separate ownership-only edges from investigative
        # evidence and are therefore unavailable, not suspicious or safe.
        return {}
    mapped = nodes.copy()
    mapped['_entity'] = np.where(
        mapped.identity_status.eq('confirmed') & mapped.canonical_player_id.notna(),
        'PLAYER:' + mapped.canonical_player_id.astype(str),
        'SOURCE:' + mapped.node_id.astype(str),
    )
    return mapped.groupby('_entity').maximum_investigation_strength.max().clip(0, 1).to_dict()


def _recent_share_signal(recent, monthly) -> float:
    recent, monthly = _number(recent), _number(monthly)
    if np.isnan(recent) or np.isnan(monthly) or monthly <= 0:
        return np.nan
    # 7/30 is the stable-rate expectation; the remaining range maps to 0-1.
    return _unit((recent / monthly - 7 / 30) / (1 - 7 / 30))


def _signals(row: pd.Series, transaction: dict, network_signal: float):
    active = _number(row.get('active_channels_30d'))
    difference = abs(_number(row.get('channel_unified_deposit_difference_30d')))
    unified_deposits = _number(row.get('unified_deposit_amount_30d'))
    participation_change = _number(row.get('channel_participation_change'))
    cross_inconsistency = np.nan
    if np.isfinite(active) and np.isfinite(difference) and np.isfinite(unified_deposits):
        gap = 0.0 if unified_deposits <= 0 else np.clip(difference / unified_deposits, 0, 1)
        multi_channel = np.clip((active - 1) / 2, 0, 1)
        change = 0.0 if np.isnan(participation_change) else np.clip(participation_change / 2, 0, 1)
        cross_inconsistency = float(.7 * gap * multi_channel + .3 * change)

    acceleration = _number(row.get('unified_spending_acceleration'))
    session_change = _number(row.get('unified_session_frequency_change'))
    peer_change = _number(row.get('peer_deposit_robust_z'))
    loss_reliability = row.get('unified_net_loss_reliable_30d', False)
    loss_is_reliable = pd.notna(loss_reliability) and bool(loss_reliability)
    aml = {
        'transaction_anomaly': _unit(_number(row.get('anomaly_severity')) / 100),
        'unusual_fund_movement': _unit(transaction.get('unusual_fund_movement')),
        'network_relationships': _unit(network_signal),
        'geographic_anomaly': _unit(transaction.get('geographic_anomaly')),
        'behavioral_change': _unit(_number(row.get('behavioral_change_score')) / 100),
        'cross_channel_inconsistency': _unit(cross_inconsistency),
    }
    safer_gambling = {
        'deposit_escalation': _unit(max(acceleration, 0) / 3)
        if np.isfinite(acceleration) else np.nan,
        'observed_net_loss_increase': _recent_share_signal(
            row.get('unified_net_loss_7d'), row.get('unified_net_loss_30d')
        ) if loss_is_reliable else np.nan,
        'session_frequency_increase': _unit(max(session_change, 0) / 3)
        if np.isfinite(session_change) else np.nan,
        'gambling_intensity_change': _recent_share_signal(
            row.get('unified_wager_volume_7d'), row.get('unified_wager_volume_30d')
        ),
        'historical_pattern_change': _unit(abs(peer_change) / 5)
        if np.isfinite(peer_change) else np.nan,
    }
    return aml, safer_gambling


def _weighted_index(signals: Mapping[str, float], weights: Mapping[str, float]):
    available = [name for name, value in signals.items() if np.isfinite(value)]
    available_weight = sum(weights[name] for name in available)
    configured_weight = sum(weights.values())
    ledger = {}
    score = 0.0
    for name, value in signals.items():
        is_available = name in available
        normalized_weight = weights[name] / available_weight if is_available else None
        contribution = float(value * normalized_weight * 100) if is_available else None
        if contribution is not None:
            score += contribution
        ledger[name] = {
            'available': is_available,
            'signal_value': float(value) if np.isfinite(value) else None,
            'configured_weight': float(weights[name]),
            'normalized_weight': normalized_weight,
            'score_contribution': contribution,
        }
    ledger['score_available'] = bool(available_weight > 0)
    ledger['available_weight'] = float(available_weight)
    ledger['evidence_weight_coverage'] = float(available_weight / configured_weight)
    ledger['missing_signals'] = [name for name in signals if name not in available]
    return float(np.clip(score, 0, 100)), ledger


def _graded_signal_frame(records: list[dict[str, float]]) -> pd.DataFrame:
    """Blend capped raw strength with population percentile for useful gradation."""
    frame = pd.DataFrame(records, dtype=float)
    for column in frame:
        raw = frame[column]
        valid = raw.notna()
        percentile = raw[valid].rank(method='average', pct=True)
        graded = (.70 * raw[valid] + .30 * percentile).clip(upper=.98)
        graded = graded.where(raw[valid].gt(0), 0.0)
        frame.loc[valid, column] = graded
    return frame


def _coverage(row: pd.Series) -> dict[str, object]:
    def clean(name):
        value = _number(row.get(name))
        return None if np.isnan(value) else float(value)
    return {
        'channel_coverage_30d': clean('channel_coverage_30d'),
        'observed_record_fraction_30d': clean('observed_record_fraction_30d'),
        'missing_field_fraction_30d': clean('missing_field_fraction_30d'),
        'known_unobserved_records_30d': clean('known_unobserved_records_30d'),
        'data_sufficiency_status': row.get('data_sufficiency_status', 'unknown'),
    }


def _risk_band(priority: float, config: RiskScoringConfig) -> str:
    if priority >= config.critical_cutoff:
        return 'critical'
    if priority >= config.high_cutoff:
        return 'high'
    if priority >= config.moderate_cutoff:
        return 'moderate'
    return 'low'


def prioritize_investigations(scored: pd.DataFrame, review_capacity: int) -> pd.DataFrame:
    """Balance the two independent domains within a fixed review budget."""
    if review_capacity < 1:
        raise ValueError('review_capacity must be positive')
    out = scored.copy()
    out['_priority_index'] = out[['aml_fraud_score', 'safer_gambling_score']].max(axis=1)
    out['_secondary_index'] = out[['aml_fraud_score', 'safer_gambling_score']].min(axis=1)
    out['_aml_rank'] = out.aml_fraud_score.rank(method='first', ascending=False)
    out['_safer_rank'] = out.safer_gambling_score.rank(method='first', ascending=False)
    out['_domain_rank'] = out[['_aml_rank', '_safer_rank']].min(axis=1)
    out['primary_risk_domain'] = np.where(
        out.aml_fraud_score.ge(out.safer_gambling_score), 'AML / Fraud', 'Safer Gambling'
    )
    out = out.sort_values(
        ['_domain_rank', '_priority_index', '_secondary_index', 'activity_entity_id'],
        ascending=[True, False, False, True], kind='stable',
    ).reset_index(drop=True)
    out['priority_rank'] = np.arange(1, len(out) + 1, dtype=int)
    out['selected_for_review'] = out.priority_rank.le(review_capacity)
    return out.drop(columns=[
        '_priority_index', '_secondary_index', '_aml_rank', '_safer_rank', '_domain_rank'
    ])


def score_risk_indices(
    feature_history: pd.DataFrame,
    model_outputs: pd.DataFrame,
    *,
    transactions: pd.DataFrame | None = None,
    network_nodes: pd.DataFrame | None = None,
    config: RiskScoringConfig = DEFAULT_RISK_SCORING_CONFIG,
    review_capacity: int | None = None,
) -> pd.DataFrame:
    """Return separate explainable AML/Fraud and Safer Gambling indices.

    Inputs are the causal Phase 4 features, Phase 5 inference output, and
    optional normalized transactions and Phase 6 nodes. Scenario labels are
    rejected. Observed transaction locations may be compared with the same
    account's past activity; age and registered/home geography are never used.
    """
    _reject_forbidden_columns(feature_history, 'feature_history', profiles=True)
    _reject_forbidden_columns(model_outputs, 'model_outputs', profiles=True)
    _reject_forbidden_columns(transactions, 'transactions')
    _reject_forbidden_columns(network_nodes, 'network_nodes', profiles=True)
    if 'activity_entity_id' not in model_outputs:
        raise ValueError('model_outputs must contain activity_entity_id')

    latest = _latest_features(feature_history)
    model = (
        model_outputs.sort_values('timestamp').groupby(
            'activity_entity_id', dropna=False
        ).tail(1)
        if 'timestamp' in model_outputs
        else model_outputs.drop_duplicates('activity_entity_id', keep='last')
    )
    model_columns = [
        column for column in model.columns
        if column == 'activity_entity_id' or column not in latest.columns
    ]
    current = latest.merge(
        model[model_columns], on='activity_entity_id', how='left', validate='one_to_one'
    )
    transaction_context = _transaction_context(
        transactions, config.maximum_supporting_transactions
    )
    network_context = _network_context(network_nodes)

    contexts = []
    raw_aml, raw_safer = [], []
    for _, row in current.iterrows():
        entity = str(row.activity_entity_id)
        transaction = transaction_context.get(entity, {})
        aml_signals, safer_signals = _signals(
            row, transaction, network_context.get(entity, np.nan)
        )
        contexts.append((row, transaction))
        raw_aml.append(aml_signals)
        raw_safer.append(safer_signals)

    graded_aml = _graded_signal_frame(raw_aml)
    graded_safer = _graded_signal_frame(raw_safer)
    rows = []
    for position, (row, transaction) in enumerate(contexts):
        aml_signals = graded_aml.iloc[position].to_dict()
        safer_signals = graded_safer.iloc[position].to_dict()
        aml_score, aml_ledger = _weighted_index(aml_signals, config.aml_weights)
        safer_score, safer_ledger = _weighted_index(
            safer_signals, config.safer_gambling_weights
        )
        identity_confidence = _number(row.get('identity_confidence'))
        combined_priority = max(aml_score, safer_score)
        rows.append({
            'player_id': row.get('canonical_player_id', row.get('player_id')),
            'activity_entity_id': row.activity_entity_id,
            'identity_status': row.get('identity_status', 'unknown'),
            'aml_fraud_score': aml_score,
            'safer_gambling_score': safer_score,
            'risk_band': _risk_band(combined_priority, config),
            'signal_contributions': {
                'aml_fraud': aml_ledger,
                'safer_gambling': safer_ledger,
            },
            'data_coverage': _coverage(row),
            'identity_confidence': None if np.isnan(identity_confidence) else identity_confidence,
            'supporting_transaction_ids': transaction.get('supporting_transaction_ids', []),
        })
    result = pd.DataFrame(rows)
    capacity = config.review_capacity if review_capacity is None else review_capacity
    return prioritize_investigations(result, capacity)


# Backwards-compatible starter dashboard API.
def score(features, config: ModelConfig = DEFAULT_MODEL_CONFIG):
    """Convert legacy demonstration outputs to review indices."""
    outputs = predict(features, config)
    out = features.merge(outputs, on='player_id', validate='one_to_one')
    out['aml_fraud_score'] = np.rint(100 * out.aml_fraud_risk).astype(int)
    out['safer_gambling_score'] = np.rint(100 * out.safer_gambling_risk).astype(int)
    out['priority'] = out[['aml_fraud_score', 'safer_gambling_score']].max(axis=1)
    return out.sort_values('priority', ascending=False).reset_index(drop=True)


def explain(row, config: ModelConfig = DEFAULT_MODEL_CONFIG):
    """Build evidence, counter-evidence, and a legacy review recommendation."""
    evidence, uncertainty = [], []
    if row.change_signal > .3:
        evidence.append(
            f'Recent weekly deposits are {row.change_ratio:.1f}x the observed '
            'historical weekly baseline.'
        )
    else:
        uncertainty.append('No substantial deposit acceleration was detected in available records.')
    if row.network_signal > .5:
        evidence.append('Payment token appears on multiple accounts in the recent observed data.')
    else:
        uncertainty.append('No shared payment-token connection detected by this configuration.')
    if row.geo_signal > .3:
        evidence.append(
            f'{row.geo_signal:.0%} of recent observed transactions originated outside the home region.'
        )
    else:
        uncertainty.append('Most observed transactions match the registered home region.')
    if row.loss_signal > .5:
        evidence.append('Recent observed net losses are elevated relative to the historical deposit baseline.')
    if row.data_quality == 'limited':
        uncertainty.append(
            'Limited recent records make this assessment uncertain; absence of evidence is not evidence of safety.'
        )
    else:
        uncertainty.append(
            'Available observations cover at least three recent weeks; unobserved channel activity may still exist.'
        )
    if row.priority >= config.high_priority_cutoff:
        action = 'Prioritize human review of supporting records; do not take automatic adverse action.'
    elif row.priority >= config.triage_cutoff:
        action = 'Queue for analyst triage and additional context.'
    else:
        action = 'Continue routine monitoring; review if new evidence appears.'
    return evidence or ['No strong indicator was detected in available records.'], uncertainty, action
