"""Transparent models with distinct AML/fraud and safer-gambling outputs.

These deterministic weighted models are an explainable hackathon baseline. They
are not fitted estimators, calibrated probabilities, or production decisions.
"""

import pandas as pd

from configuration import DEFAULT_MODEL_CONFIG, ModelConfig


MODEL_FEATURES = (
    'change_signal',
    'network_signal',
    'geo_signal',
    'cross_signal',
    'loss_signal',
)


def predict(features: pd.DataFrame, config: ModelConfig = DEFAULT_MODEL_CONFIG) -> pd.DataFrame:
    """Produce independent continuous outputs in the closed interval [0, 1]."""
    missing = set(MODEL_FEATURES).difference(features.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    outputs = pd.DataFrame({'player_id': features['player_id']})
    outputs['aml_fraud_risk'] = (
        config.aml_change_weight * features.change_signal
        + config.aml_network_weight * features.network_signal
        + config.aml_geo_weight * features.geo_signal
        + config.aml_cross_channel_weight * features.cross_signal
    ).clip(0, 1)
    outputs['safer_gambling_risk'] = (
        config.gambling_change_weight * features.change_signal
        + config.gambling_loss_weight * features.loss_signal
    ).clip(0, 1)
    return outputs
