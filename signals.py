"""Leakage-conscious demonstration signals, with a historical vs recent split."""
import numpy as np
import pandas as pd

from configuration import DEFAULT_FEATURE_CONFIG, FeatureConfig


GROUND_TRUTH_COLUMNS = frozenset({
    'scenario', 'injected_risk', 'network_group', 'scenario_type',
    'start_time', 'end_time', 'target_risk_category', 'ground_truth_label',
})


def compute(
    profiles,
    transactions,
    cross_channel=True,
    hardened=True,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
):
    """Create operational features without accepting synthetic labels."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(profiles.columns)
    if leaked:
        raise ValueError(f"Ground-truth columns are not valid model inputs: {sorted(leaked)}")
    tx = transactions[transactions.observed].copy()
    if not cross_channel:
        tx = tx[tx.channel=='online_casino']
    split = int(transactions.week.max()) - (config.recent_weeks - 1)
    old = tx[tx.week < split]
    recent = tx[tx.week >= split]
    historic = old.groupby('player_id').agg(old_deposit=('deposit','sum'),old_weeks=('week','nunique'))
    now = recent.groupby('player_id').agg(recent_deposit=('deposit','sum'),recent_weeks=('week','nunique'), recent_loss=('net_loss','sum'),regions=('region','nunique'), channels=('channel','nunique'))
    df = profiles.set_index('player_id').join(historic).join(now)
    for c in ['old_deposit','old_weeks','recent_deposit','recent_weeks','recent_loss','regions','channels']:
        df[c] = df[c].fillna(0)
    df['old_weekly'] = df.old_deposit / np.maximum(df.old_weeks,1)
    df['recent_weekly'] = df.recent_deposit / np.maximum(df.recent_weeks,1)
    df['change_ratio'] = df.recent_weekly / np.maximum(df.old_weekly, config.minimum_historical_weekly)
    df['change_signal'] = np.clip((df.change_ratio-1.5)/3.5,0,1)
    df['loss_signal'] = np.clip(df.recent_loss / np.maximum(df.old_weekly*4,100) / 3,0,1)
    df['geo_signal'] = 0.
    outside = recent.merge(profiles[['player_id','region']],on='player_id',suffixes=('','_home'))
    if len(outside):
        mismatch = outside.assign(mismatch=(outside.region!=outside.region_home).astype(float)).groupby('player_id').mismatch.mean()
        df.loc[mismatch.index,'geo_signal'] = mismatch
    df['network_signal'] = 0.
    if hardened and len(recent):
        shared = recent.groupby('payment_token').player_id.nunique()
        suspicious_tokens = set(shared[shared>=2].index)
        linked = df.index.intersection(
            recent[recent.payment_token.isin(suspicious_tokens)].player_id.unique()
        )
        df.loc[linked,'network_signal'] = 1.
    # cross-channel fragmentation: score escalation in total combined behavior
    df['cross_signal'] = np.clip((df.change_ratio-1.25)/2.75,0,1) if cross_channel else 0.
    df['coverage'] = df.channels / config.expected_channels
    df['data_quality'] = np.where(df.recent_weeks >= config.minimum_quality_weeks, 'sufficient', 'limited')
    return df.reset_index()
