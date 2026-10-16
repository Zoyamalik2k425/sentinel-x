"""Fixed alert-budget comparisons; synthetic labels only."""
import pandas as pd
from signals import compute
from scoring import score

def evaluate(profiles, tx, ground_truth, k=40):
    """Evaluate only against held synthetic labels at a fixed alert budget."""
    rows=[]
    for name,cross,hardened in [('Single-channel V1',False,False),('Cross-channel V1',True,False),('Cross-channel V2',True,True)]:
        scored=score(compute(profiles,tx,cross,hardened))
        k0=min(k,len(scored))
        flagged=set(scored.head(k0).player_id)
        risky=set(ground_truth.loc[ground_truth.ground_truth_label,'player_id'])
        tp=len(flagged&risky)
        rows.append(dict(model=name,alerts=k0,true_positives=tp,precision_at_k=tp/max(k0,1),recall_at_k=tp/max(len(risky),1)))
    return pd.DataFrame(rows)

def group_audit(scored, field='age_band',k=40):
    frame=scored.copy()
    frame['flagged']=False
    frame.loc[frame.head(min(k,len(frame))).index,'flagged']=True
    return frame.groupby(field,dropna=False).agg(players=('player_id','count'),flags=('flagged','sum'),flag_rate=('flagged','mean')).reset_index()
