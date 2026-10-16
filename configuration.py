"""Central, reproducible settings for the Sentinel X prototype."""

from dataclasses import dataclass


RANDOM_SEED = 42


@dataclass(frozen=True)
class DataConfig:
    """Synthetic data settings; no production data is represented."""

    n_players: int = 2_000
    history_days: int = 182
    target_events: int = 100_000
    seed: int = RANDOM_SEED
    start_date: str = "2026-01-01"
    risky_scenario_fraction: float = 0.03


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for leakage-conscious historical and recent feature windows."""

    recent_weeks: int = 5
    minimum_historical_weekly: float = 20.0
    minimum_quality_weeks: int = 3
    expected_channels: int = 3


@dataclass(frozen=True)
class ModelConfig:
    """Transparent model weights and risk-index cutoffs."""

    aml_change_weight: float = 0.30
    aml_network_weight: float = 0.40
    aml_geo_weight: float = 0.20
    aml_cross_channel_weight: float = 0.10
    gambling_change_weight: float = 0.50
    gambling_loss_weight: float = 0.50
    high_priority_cutoff: int = 60
    triage_cutoff: int = 30


DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
