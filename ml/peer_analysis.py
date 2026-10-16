"""Historical behavioral clustering and robust peer deviation scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DEFAULT_CLUSTER_FEATURES = (
    'unified_historical_median_spending', 'unified_historical_spending_mad',
    'unified_observed_events_30d', 'unified_session_frequency_30d',
    'unified_wager_volume_30d', 'active_channels_30d',
    'unified_behavioral_stability', 'account_tenure_days',
)
DEFAULT_COMPARISON_FEATURES = (
    'unified_deposit_amount_30d', 'unified_wager_volume_30d',
    'unified_withdrawal_frequency_30d', 'unified_net_loss_30d',
    'unified_session_frequency_30d', 'unified_distinct_devices_30d',
    'unified_distinct_payment_instruments_30d',
)


class BehavioralPeerAnalyzer:
    """Group entities by behavior and compare them with training-only peers."""

    def __init__(
        self,
        cluster_features=DEFAULT_CLUSTER_FEATURES,
        comparison_features=DEFAULT_COMPARISON_FEATURES,
        n_clusters: int = 6,
        random_state: int = 42,
    ):
        self.cluster_features = tuple(cluster_features)
        self.comparison_features = tuple(comparison_features)
        forbidden = {'age', 'age_band', 'region'}
        used = set(self.cluster_features) | set(self.comparison_features)
        if forbidden.intersection(used):
            raise ValueError('Age and region cannot be peer risk features')
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', RobustScaler()),
        ])
        self.model = None
        self.distributions = {}

    def _matrix(self, frame, columns):
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f'Missing peer features: {sorted(missing)}')
        return frame.loc[:, columns].apply(pd.to_numeric, errors='coerce')

    def fit(self, historical_training_data: pd.DataFrame):
        if len(historical_training_data) < 4:
            raise ValueError('Peer analysis requires at least four historical entities')
        cluster_matrix = self._matrix(historical_training_data, self.cluster_features)
        transformed = self.preprocessor.fit_transform(cluster_matrix)
        clusters = min(self.n_clusters, max(2, len(historical_training_data) // 3))
        self.model = KMeans(n_clusters=clusters, n_init=10, random_state=self.random_state)
        labels = self.model.fit_predict(transformed)
        comparison = self._matrix(historical_training_data, self.comparison_features)
        for cluster in range(clusters):
            subset = comparison.loc[labels == cluster]
            medians = subset.median(skipna=True).fillna(0.0)
            mad = subset.sub(medians).abs().median(skipna=True)
            fallback = comparison.quantile(.75) - comparison.quantile(.25)
            scales = (1.4826 * mad).where(mad >= 1e-9, fallback).replace(0, np.nan).fillna(1.0)
            self.distributions[cluster] = (medians, scales, len(subset))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError('BehavioralPeerAnalyzer must be fitted before inference')
        transformed = self.preprocessor.transform(self._matrix(frame, self.cluster_features))
        clusters = self.model.predict(transformed)
        comparison = self._matrix(frame, self.comparison_features)
        scores, indicators, peer_counts = [], [], []
        for position, (_, row) in enumerate(comparison.iterrows()):
            median, scale, count = self.distributions[int(clusters[position])]
            deviation = row.fillna(median).sub(median).div(scale).abs()
            maximum = float(deviation.max())
            scores.append(float(np.clip(100 * (1 - np.exp(-maximum / 3.0)), 0, 100)))
            indicators.append('|'.join(
                f'{name}:{value:.2f}' for name, value in deviation.sort_values(ascending=False).head(3).items()
            ))
            peer_counts.append(count)
        return pd.DataFrame({
            'peer_cluster': clusters,
            'peer_training_count': peer_counts,
            'peer_deviation_score': scores,
            'peer_feature_indicators': indicators,
        }, index=frame.index)
