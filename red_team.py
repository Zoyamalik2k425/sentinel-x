"""Synthetic stress tests of observable records, not instructions for evading real monitoring."""
import pandas as pd

def stress_test(tx, ground_truth, level):
    """Simulate loss of channel visibility and noisier observations for risky scenarios."""
    if level not in range(4):
        raise ValueError('level must be one of 0, 1, 2, or 3')
    out=tx.copy()
    if level==0: return out
    risky=set(ground_truth.loc[ground_truth.ground_truth_label,'player_id'])
    mask=out.player_id.isin(risky)&(out.week>=out.week.max()-4)
    if level>=1:
        out.loc[mask & (out.channel=='land_casino'),'observed']=False
    if level>=2:
        out.loc[mask & (out.channel=='sportsbook'),'observed']=False
    if level>=3:
        # Missingness stress test: selectively remove some recent records from observation.
        ids=out.index[mask & (out.channel=='online_casino')]
        out.loc[ids[::2],'observed']=False
    return out
