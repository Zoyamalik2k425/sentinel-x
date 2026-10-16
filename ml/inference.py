"""Combined inference and persistence for Sentinel X anomaly models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .anomaly_detection import IsolationForestDetector
from .behavioral_change import score_behavioral_change, score_latest_behavioral_change
from .peer_analysis import BehavioralPeerAnalyzer


@dataclass
class ModelBundle:
    anomaly_detector: IsolationForestDetector
    peer_analyzer: BehavioralPeerAnalyzer
    alert_threshold: float
    trained_through: pd.Timestamp
    validated_through: pd.Timestamp | None = None
    model_version: str = 'phase5-v1'

    def save(self, path: str | Path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path):
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError('Saved artifact is not a Sentinel X ModelBundle')
        return bundle


def _combine_indicators(row) -> str:
    candidates = []
    for source in ('anomaly_feature_indicators', 'peer_feature_indicators'):
        value = row.get(source)
        if isinstance(value, str):
            candidates.extend(value.split('|'))
    if pd.notna(row.get('behavioral_change_robust_z')):
        candidates.append(f"own_history_change:{row['behavioral_change_robust_z']:.2f}")
    seen, result = set(), []
    for item in candidates:
        name = item.split(':', 1)[0]
        if name not in seen:
            result.append(item)
            seen.add(name)
        if len(result) == 5:
            break
    return '|'.join(result)


def _sufficiency(row) -> str:
    days = row.get('unified_observation_days')
    events = row.get('unified_observed_events_30d')
    coverage = row.get('channel_coverage_30d')
    if pd.isna(days) or pd.isna(events) or days < 30 or events < 5:
        return 'limited'
    if days < 90 or events < 15 or coverage < (2 / 3):
        return 'moderate'
    return 'sufficient'


def run_inference(
    bundle: ModelBundle,
    feature_history: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    latest_change_only: bool = False,
) -> pd.DataFrame:
    """Return one current output row per activity entity using data through ``as_of``."""
    history = feature_history.copy()
    history['timestamp'] = pd.to_datetime(history.timestamp)
    as_of = pd.Timestamp(as_of) if as_of is not None else history.timestamp.max()
    history = history[history.timestamp <= as_of].sort_values(['timestamp', 'event_id'])
    if history.empty:
        return pd.DataFrame()
    change = (
        score_latest_behavioral_change(history)
        if latest_change_only else score_behavioral_change(history)
    )
    latest = history.groupby('activity_entity_id', dropna=False).tail(1).copy()
    anomaly = bundle.anomaly_detector.predict(latest)
    peer = bundle.peer_analyzer.predict(latest)
    latest = pd.concat([latest, anomaly, peer, change.loc[latest.index]], axis=1)
    latest['model_priority_index'] = latest[[
        'anomaly_severity', 'behavioral_change_score', 'peer_deviation_score'
    ]].max(axis=1)
    latest['alert'] = latest.model_priority_index >= bundle.alert_threshold
    latest['top_contributing_feature_indicators'] = latest.apply(_combine_indicators, axis=1)
    latest['data_sufficiency_status'] = latest.apply(_sufficiency, axis=1)
    latest['available_historical_context'] = latest.apply(
        lambda row: json.dumps({
            'observation_days': None if pd.isna(row.unified_observation_days) else round(float(row.unified_observation_days), 1),
            'observed_events_30d': int(row.unified_observed_events_30d),
            'active_channels_30d': int(row.active_channels_30d),
            'channel_coverage_30d': round(float(row.channel_coverage_30d), 3),
            'identity_status': row.identity_status,
            'identity_confidence': round(float(row.identity_confidence), 3),
        }),
        axis=1,
    )
    latest['player_id'] = latest.canonical_player_id
    return latest[[
        'player_id', 'activity_entity_id', 'timestamp', 'anomaly_severity',
        'behavioral_change_score', 'peer_deviation_score', 'model_priority_index',
        'top_contributing_feature_indicators', 'available_historical_context',
        'data_sufficiency_status', 'channel_coverage_30d', 'alert',
    ]].reset_index(drop=True)
