"""Risk scoring and workload prioritization for transparent model outputs."""

import numpy as np

from configuration import DEFAULT_MODEL_CONFIG, ModelConfig
from models import predict


def score(features, config: ModelConfig = DEFAULT_MODEL_CONFIG):
    """Convert separate model outputs to review indices and a priority rank."""
    outputs = predict(features, config)
    out = features.merge(outputs, on='player_id', validate='one_to_one')
    out['aml_fraud_score'] = np.rint(100 * out.aml_fraud_risk).astype(int)
    out['safer_gambling_score'] = np.rint(100 * out.safer_gambling_risk).astype(int)
    out['priority'] = out[['aml_fraud_score', 'safer_gambling_score']].max(axis=1)
    return out.sort_values('priority', ascending=False).reset_index(drop=True)


def explain(row, config: ModelConfig = DEFAULT_MODEL_CONFIG):
    """Build evidence, counter-evidence, and a human-review recommendation."""
    evidence, uncertainty = [], []
    if row.change_signal > .3:
        evidence.append(
            f"Recent weekly deposits are {row.change_ratio:.1f}× the observed "
            "historical weekly baseline."
        )
    else:
        uncertainty.append('No substantial deposit acceleration was detected in available records.')
    if row.network_signal > .5:
        evidence.append('Payment token appears on multiple accounts in the recent observed data.')
    else:
        uncertainty.append('No shared payment-token connection detected by this configuration.')
    if row.geo_signal > .3:
        evidence.append(
            f"{row.geo_signal:.0%} of recent observed transactions originated outside the home region."
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
