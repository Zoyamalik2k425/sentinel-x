"""Sentinel X: fictional, offline, cross-channel risk intelligence prototype."""

import pandas as pd
import plotly.express as px
import streamlit as st

from configuration import RANDOM_SEED
from data_gen import OPERATIONAL_DATA_SCHEMA_VERSION, OperationalDataset, generate
from red_team import stress_test
from scoring import explain, score
from signals import compute
from validate import evaluate, group_audit


st.set_page_config(page_title='Sentinel X | OLG Hackathon', layout='wide', page_icon='🛡️')
st.title('🛡️ Sentinel X')
st.caption('Cross-channel investigation intelligence • FICTIONAL SYNTHETIC DATA • Not an OLG production system')


@st.cache_data(show_spinner='Generating synthetic data...')
def load_app_data(n: int, seed: int, schema_version: int) -> OperationalDataset:
    """Load the versioned, named application contract through Streamlit's cache."""
    if schema_version != OPERATIONAL_DATA_SCHEMA_VERSION:
        raise ValueError('Unsupported operational data schema version')
    return generate(n_players=n, seed=seed)


@st.cache_data(show_spinner=False)
def model(profiles, transactions, cross_channel, hardened):
    return score(compute(profiles, transactions, cross_channel, hardened))


with st.sidebar:
    st.header('Experiment controls')
    n = st.select_slider('Synthetic players', options=[200, 500, 1000], value=500)
    seed = st.number_input('Random seed', value=RANDOM_SEED, min_value=0)
    cross = st.toggle('Combine available channels', value=True)
    hardened = st.toggle('Enable network signal (V2)', value=True)
    threshold = st.slider('Review threshold', 0, 100, 35)
    st.info('Risk indices are heuristic rankings, NOT fraud probabilities. No real player data is used.')

app_data = load_app_data(n, seed, OPERATIONAL_DATA_SCHEMA_VERSION)
profiles = app_data.profiles
transactions = app_data.transactions
ground_truth = app_data.scenario_labels
scored = model(profiles, transactions, cross, hardened)
queue = scored[scored.priority >= threshold]

t1, t2, t3, t4 = st.tabs([
    '📋 Investigation Queue', '🔎 Player 360', '🧪 Stress Test Arena', '📊 Validation & Fairness'
])

with t1:
    a, b, c, d = st.columns(4)
    a.metric('Synthetic players', len(profiles))
    b.metric('Observed records', int(transactions.observed.sum()))
    c.metric('Review queue', len(queue))
    d.metric('Limited-data profiles', int((scored.data_quality == 'limited').sum()))
    st.info('Operational scoring is isolated from synthetic scenario labels; labels are used only for offline evaluation.')
    st.dataframe(
        queue[['player_id', 'aml_fraud_score', 'safer_gambling_score', 'priority', 'data_quality', 'coverage']]
        .rename(columns={'coverage': 'recent channel coverage'}),
        width='stretch', hide_index=True,
    )
    st.plotly_chart(
        px.histogram(scored, x='priority', nbins=20, title='Distribution of heuristic priority indices'),
        width='stretch',
    )

with t2:
    pid = st.selectbox('Choose player', scored.player_id.tolist())
    row = scored.loc[scored.player_id == pid].iloc[0]
    st.subheader(f'Player {pid}')
    a, b, c = st.columns(3)
    a.metric('AML/Fraud review index', int(row.aml_fraud_score))
    b.metric('Safer gambling review index', int(row.safer_gambling_score))
    c.metric('Observed recent channels', f'{int(row.channels)} / 3')
    if row.data_quality == 'limited':
        st.warning('Limited recent observations. Missing channel data must not be interpreted as zero activity.')
    player_tx = transactions[(transactions.player_id == pid) & transactions.observed].copy()
    weekly = player_tx.groupby(['week', 'channel'], as_index=False).deposit.sum()
    st.plotly_chart(
        px.line(weekly, x='week', y='deposit', color='channel', markers=True,
                title='Observed weekly deposits by channel (CAD)'),
        width='stretch',
    )
    evidence, uncertainty, action = explain(row)
    left, right = st.columns(2)
    with left:
        st.markdown('#### Evidence supporting review')
        for item in evidence:
            st.write('• ' + item)
    with right:
        st.markdown('#### Counter-evidence / uncertainty')
        for item in uncertainty:
            st.write('• ' + item)
    st.info('**Suggested next step:** ' + action)
    if 'decisions' not in st.session_state:
        st.session_state.decisions = {}
    a, b = st.columns(2)
    if a.button('Mark for review'):
        st.session_state.decisions[pid] = 'Review'
    if b.button('Dismiss alert'):
        st.session_state.decisions[pid] = 'Dismissed'
    st.caption('Local session decision: ' + st.session_state.decisions.get(pid, 'Not reviewed'))
    st.dataframe(
        player_tx[['tx_id', 'timestamp', 'channel', 'deposit', 'wager', 'net_loss', 'region', 'payment_token']]
        .sort_values('timestamp', ascending=False),
        width='stretch', hide_index=True,
    )

with t3:
    st.subheader('Cross-channel visibility stress test')
    st.write(
        'Test the same fictional player histories as observed channel records become less complete. '
        'This tests robustness to missing information, not real-world evasion.'
    )
    k = st.slider('Investigation capacity (top K)', 10, min(150, n), min(40, n), key='arena_k')
    if st.button('Run all four stress levels', type='primary'):
        result = []
        for level in range(4):
            altered = stress_test(transactions, ground_truth, level)
            report = evaluate(profiles, altered, ground_truth, k)
            report['stress_level'] = level
            result.append(report)
        st.session_state['arena_results'] = pd.concat(result, ignore_index=True)
    if 'arena_results' in st.session_state:
        results = st.session_state.arena_results
        st.plotly_chart(
            px.line(results, x='stress_level', y='recall_at_k', color='model', markers=True,
                    title='Recall at fixed investigator capacity'),
            width='stretch',
        )
        st.dataframe(results, width='stretch', hide_index=True)
        st.caption('Results are computed, not predetermined. Hardening may or may not improve recall under missingness.')

with t4:
    k = st.slider('Top K alert budget', 10, min(150, n), min(40, n), key='eval_k')
    report = evaluate(profiles, transactions, ground_truth, k)
    st.subheader('Held scenario labels: demonstration evaluation')
    st.caption(f"Synthetic risky scenarios in evaluation set: {int(ground_truth.ground_truth_label.sum())}")
    st.dataframe(report, width='stretch', hide_index=True)
    st.plotly_chart(
        px.bar(report, x='model', y=['precision_at_k', 'recall_at_k'], barmode='group',
               title='Fixed-budget model comparison'),
        width='stretch',
    )
    st.subheader('Flag-rate audit (descriptive, not a fairness guarantee)')
    field = st.radio('Audit group', options=['age_band', 'region'], horizontal=True)
    audit = group_audit(scored, field, k)
    st.dataframe(audit, width='stretch', hide_index=True)
    st.plotly_chart(
        px.bar(audit, x=field, y='flag_rate', title='Observed top-K flag rate by group'),
        width='stretch',
    )
    st.caption(
        'Synthetic labels, data-generation assumptions, group composition, and missingness limit generalization. '
        'Group flag-rate differences alone do not establish discrimination.'
    )
