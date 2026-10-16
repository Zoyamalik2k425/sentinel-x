"""Isolation Forest anomaly ranking with robust local preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DEFAULT_ANOMALY_FEATURES = (
    'unified_deposit_frequency_7d', 'unified_deposit_frequency_30d',
    'unified_deposit_amount_7d', 'unified_deposit_amount_30d',
    'unified_withdrawal_frequency_30d', 'unified_wager_volume_30d',
    'unified_net_loss_30d', 'unified_session_frequency_30d',
    'unified_spending_acceleration', 'unified_behavioral_stability',
    'unified_distinct_devices_30d', 'unified_distinct_payment_instruments_30d',
    'active_channels_30d', 'channel_participation_change',
    'channel_coverage_30d', 'observed_record_fraction_30d',
    'account_tenure_days',
)
FORBIDDEN_FEATURES = frozenset({
    'age', 'age_band', 'region', 'scenario', 'scenario_type',
    'ground_truth_label', 'injected_risk', 'target_risk_category',
})


class IsolationForestDetector:
    """Unsupervised anomaly detector returning a non-probabilistic 0–100 index."""

    def __init__(
        self,
        feature_columns=DEFAULT_ANOMALY_FEATURES,
        contamination: float = .08,
        random_state: int = 42,
        n_estimators: int = 250,
    ):
        self.feature_columns = tuple(feature_columns)
        forbidden = FORBIDDEN_FEATURES.intersection(self.feature_columns)
        if forbidden:
            raise ValueError(f'Forbidden model features: {sorted(forbidden)}')
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('scaler', RobustScaler()),
            ('model', IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._training_raw_scores = None
        self._feature_medians = None
        self._feature_scales = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f'Missing anomaly features: {sorted(missing)}')
        return frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors='coerce')

    def fit(self, historical_training_data: pd.DataFrame):
        matrix = self._matrix(historical_training_data)
        if len(matrix) < 8:
            raise ValueError('Isolation Forest requires at least eight historical entities')
        self.pipeline.fit(matrix)
        raw = -self.pipeline.decision_function(matrix)
        self._training_raw_scores = np.sort(raw)
        self._feature_medians = matrix.median(skipna=True).fillna(0.0)
        q75 = matrix.quantile(.75)
        q25 = matrix.quantile(.25)
        self._feature_scales = (q75 - q25).replace(0, np.nan).fillna(1.0)
        return self

    def _check_fitted(self):
        if self._training_raw_scores is None:
            raise RuntimeError('IsolationForestDetector must be fitted before inference')

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return severity percentile and approximate feature indicators."""
        self._check_fitted()
        matrix = self._matrix(frame)
        raw = -self.pipeline.decision_function(matrix)
        ranks = np.searchsorted(self._training_raw_scores, raw, side='right')
        severity = np.clip(100 * ranks / len(self._training_raw_scores), 0, 100)
        deviations = (
            matrix.fillna(self._feature_medians) - self._feature_medians
        ).abs().div(self._feature_scales)
        indicators = []
        for _, row in deviations.iterrows():
            top = row.sort_values(ascending=False).head(3)
            indicators.append('|'.join(f'{name}:{value:.2f}' for name, value in top.items()))
        return pd.DataFrame({
            'anomaly_severity': severity,
            'anomaly_raw_score': raw,
            'anomaly_feature_indicators': indicators,
        }, index=frame.index)
