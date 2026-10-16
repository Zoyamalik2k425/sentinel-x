"""Sentinel X professional investigator dashboard using fictional local data."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configuration import RANDOM_SEED
from cross_channel import CrossChannelResult, integrate_channels
from dashboard_metrics import (
    evaluate_at_budget,
    fairness_audit,
    performance_by_channel,
    performance_by_scenario,
    ranked_queue,
)
from data_gen import SyntheticGamingData, generate_dataset, to_operational_events
from dashboard_features import build_dashboard_feature_history
from investigator_copilot import (
    STORE_KEY,
    SIGNAL_LABELS,
    build_investigation_case,
    escape_currency_markdown,
    render_investigator_controls,
)
from ml.inference import ModelBundle, run_inference
from ml.train import chronological_split, fit_unsupervised_models
from network_analysis import NetworkAnalysisResult, build_relationship_graph
from red_team import BEHAVIOR_LEVELS, run_adversarial_arena, stress_test
from scoring import score_risk_indices
from validate import detector_scores, evaluate as evaluate_legacy_models, evaluate_ranked_scores


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_UI_PLAYERS = int(os.environ.get('SENTINEL_X_DEFAULT_PLAYERS', '2000'))
MODEL_PATH = APP_ROOT / 'artifacts' / 'phase5_models.joblib'
CHANNEL_COLORS = {
    'online_casino': '#2f81f7',
    'sportsbook': '#22d3ee',
    'land_casino': '#a78bfa',
}
RISK_COLORS = {
    'low': '#38bdf8', 'moderate': '#fbbf24', 'high': '#fb7185', 'critical': '#f43f5e',
}


@dataclass(frozen=True)
class DashboardData:
    synthetic: SyntheticGamingData
    integrated: CrossChannelResult
    events: pd.DataFrame
    features: pd.DataFrame
    model_outputs: pd.DataFrame
    network: NetworkAnalysisResult
    scored: pd.DataFrame
    operational_events: pd.DataFrame
    model_version: str


st.set_page_config(
    page_title='Sentinel X | Investigator Intelligence',
    page_icon='X',
    layout='wide',
    initial_sidebar_state='expanded',
)


def _inject_theme():
    st.markdown(
        """
        <style>
        :root { --sx-navy:#071426; --sx-panel:#0d2038; --sx-blue:#2f81f7; --sx-cyan:#22d3ee; }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 82% 0%, #12315a 0%, #071426 34%, #050d19 100%);
            color: #e8f1ff;
        }
        [data-testid="stHeader"] { background: rgba(5,13,25,.82); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08172a 0%, #06101e 100%);
            border-right: 1px solid rgba(91,143,249,.18);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(16,42,73,.96), rgba(9,27,49,.96));
            border: 1px solid rgba(71,134,224,.24); border-radius: 14px;
            padding: 16px 18px; box-shadow: 0 12px 30px rgba(0,0,0,.18);
        }
        [data-testid="stMetricLabel"] { color: #b8cbe3; }
        [data-testid="stMetricLabel"] p { font-size:.76rem; line-height:1.05; }
        [data-testid="stMetricValue"] { color: #f4f8ff; }
        div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(71,134,224,.18); border-radius: 14px; overflow: hidden;
        }
        .sx-kicker { color:#66d9ff; font-size:.74rem; font-weight:700; letter-spacing:.16em; }
        .sx-title { font-size:2rem; font-weight:760; margin:.15rem 0; color:#f7fbff; }
        .sx-subtitle { color:#9fb6d4; margin-bottom:1.25rem; }
        .sx-card {
            background:linear-gradient(145deg,rgba(16,42,73,.9),rgba(9,27,49,.9));
            border:1px solid rgba(71,134,224,.22); border-radius:14px; padding:16px 18px;
        }
        .sx-pill { display:inline-block; padding:4px 9px; border-radius:999px;
            background:rgba(47,129,247,.16); color:#8fc5ff; font-size:.78rem; margin-right:6px; }
        h1,h2,h3,h4 { color:#f3f8ff !important; letter-spacing:-.015em; }
        p,li,[data-testid="stCaptionContainer"] { color:#b9c9dd; }
        .stButton > button { border-radius:9px; border:1px solid rgba(80,146,238,.45); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(kicker: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="sx-kicker">{kicker}</div><div class="sx-title">{title}</div>'
        f'<div class="sx-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def _show_table(frame: pd.DataFrame, **kwargs):
    """Render tables with readable labels while preserving numeric formats."""
    overrides = kwargs.pop('column_config', {})
    labels = {
        column: st.column_config.Column(
            str(column).replace('_', ' ').replace('aml', 'AML').title()
            .replace('Aml', 'AML').replace('Id', 'ID')
        )
        for column in frame.columns
    }
    labels.update(overrides)
    return st.dataframe(frame, column_config=labels, **kwargs)


def _events_target(player_count: int) -> int:
    # The full 2,000-player mode meets the Phase 2 80k floor. Smaller UI modes
    # use 20 events/player to keep local iteration fast.
    return player_count * 10


def _fallback_bundle(features: pd.DataFrame, seed: int) -> ModelBundle:
    splits = chronological_split(features)
    anomaly, peers = fit_unsupervised_models(splits.train, random_state=seed)
    return ModelBundle(
        anomaly_detector=anomaly,
        peer_analyzer=peers,
        alert_threshold=75.0,
        trained_through=splits.train_end,
        validated_through=None,
        model_version='phase9-local-fallback',
    )


def _build_dashboard_data(player_count: int, seed: int, cross_channel: bool) -> DashboardData:
    """Build one complete operational dashboard payload without held-label leakage."""
    synthetic = generate_dataset(
        n_players=player_count,
        history_days=182,
        target_events=_events_target(player_count),
        seed=seed,
    )
    integrated = integrate_channels(
        synthetic.players,
        synthetic.online_casino,
        synthetic.sports_betting,
        synthetic.land_based,
        synthetic.start_time,
        synthetic.end_time,
    )
    events = integrated.events.copy()
    if not cross_channel:
        events = events[events.channel.eq('online_casino')].copy()
    features = build_dashboard_feature_history(events, synthetic.players)
    bundle = ModelBundle.load(MODEL_PATH) if MODEL_PATH.exists() else _fallback_bundle(features, seed)
    outputs = run_inference(bundle, features, latest_change_only=True)
    network = build_relationship_graph(events, maximum_plot_nodes=260)
    scored = score_risk_indices(
        features,
        outputs,
        transactions=events,
        network_nodes=network.nodes,
        review_capacity=100,
    )
    latest_activity = events.groupby('activity_entity_id').timestamp.max()
    scored['latest_activity'] = scored.activity_entity_id.map(latest_activity)
    return DashboardData(
        synthetic=synthetic,
        integrated=integrated,
        events=events,
        features=features,
        model_outputs=outputs,
        network=network,
        scored=scored,
        operational_events=to_operational_events(synthetic),
        model_version=bundle.model_version,
    )


@st.cache_resource(show_spinner='Preparing cross-channel intelligence...')
def prepare_dashboard(player_count: int, seed: int, cross_channel: bool) -> DashboardData:
    """Return the cached, fully populated dashboard payload."""
    return _build_dashboard_data(player_count, seed, cross_channel)


@st.cache_data(show_spinner='Running defensive red-team experiments...')
def run_cached_arena(player_count: int, seed: int, budget: int):
    return run_adversarial_arena(
        n_players=player_count,
        target_events=_events_target(player_count),
        development_seed=seed + 1_701,
        evaluation_seed=seed,
        alert_budget=budget,
    )


def _risk_ledger_frame(row: pd.Series) -> pd.DataFrame:
    rows = []
    ledger = row.signal_contributions
    for category in ('aml_fraud', 'safer_gambling'):
        for signal, detail in ledger[category].items():
            if signal not in SIGNAL_LABELS or not isinstance(detail, dict):
                continue
            rows.append({
                'risk_domain': 'AML / Fraud' if category == 'aml_fraud' else 'Safer Gambling',
                'signal': SIGNAL_LABELS[signal],
                'available': detail['available'],
                'signal_value': detail['signal_value'],
                'score_contribution': detail['score_contribution'],
            })
    return pd.DataFrame(rows)


def _network_figure(network: NetworkAnalysisResult) -> go.Figure:
    nodes, edges = network.plotly_nodes, network.plotly_edges
    edge_x, edge_y = [], []
    for edge in edges.itertuples():
        edge_x.extend([edge.x0, edge.x1, None])
        edge_y.extend([edge.y0, edge.y1, None])
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines', hoverinfo='skip',
        line={'width': .8, 'color': 'rgba(110,155,210,.28)'}, name='Relationships',
    ))
    for channel, group in nodes.groupby('channel'):
        figure.add_trace(go.Scatter(
            x=group.x, y=group.y, mode='markers', name=channel.replace('_', ' ').title(),
            text=group.hover_text, hovertemplate='%{text}<extra></extra>',
            marker={
                'size': np.clip(7 + group.degree.to_numpy() * 2, 7, 22),
                'color': CHANNEL_COLORS.get(channel, '#94a3b8'),
                'line': {
                    'width': np.where(group.potentially_unusual_cluster, 2.5, .5),
                    'color': np.where(group.potentially_unusual_cluster, '#fb7185', '#dbeafe'),
                },
            },
        ))
    figure.update_layout(
        height=610, showlegend=True, hovermode='closest',
        margin={'l': 0, 'r': 0, 't': 15, 'b': 0},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#c7d8ed'},
        xaxis={'visible': False}, yaxis={'visible': False},
        legend={'orientation': 'h', 'y': 1.02},
    )
    return figure


def _overview(data: DashboardData, queue: pd.DataFrame, budget: int):
    _page_header(
        'COMMAND CENTRE', 'Overview',
        'Cross-channel risk posture from fictional, locally generated activity.',
    )
    observed = data.events[data.events.observed.fillna(False)]
    high_aml = int(queue.aml_fraud_score.ge(60).sum())
    high_sg = int(queue.safer_gambling_score.ge(60).sum())
    columns = st.columns(5)
    columns[0].metric('Synthetic players', f'{len(data.synthetic.players):,}')
    columns[1].metric('Observed transactions', f'{len(observed):,}')
    columns[2].metric('Active channels', int(observed.channel.nunique()))
    columns[3].metric('High-priority AML', f'{high_aml:,}')
    columns[4].metric('High-priority safer gambling', f'{high_sg:,}')

    left, right = st.columns([1.15, .85])
    with left:
        st.subheader('Risk distribution')
        risk = queue[['display_id', 'aml_fraud_score', 'safer_gambling_score']].melt(
            id_vars='display_id', var_name='risk_domain', value_name='score'
        )
        figure = px.histogram(
            risk, x='score', color='risk_domain', nbins=20, barmode='overlay',
            color_discrete_map={
                'aml_fraud_score': '#2f81f7', 'safer_gambling_score': '#22d3ee',
            },
        )
        figure.update_layout(
            height=360, margin=dict(l=10, r=10, t=15, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', legend_title_text='', xaxis_title='Review index (0-100)',
        )
        st.plotly_chart(figure, width='stretch')
    with right:
        st.subheader('Cross-channel activity')
        summary = observed.groupby('channel').agg(
            observed_events=('event_id', 'count'),
            active_entities=('activity_entity_id', 'nunique'),
            observed_deposits=('deposit_amount', 'sum'),
            observed_wagers=('wager_amount', lambda values: values.sum(min_count=1)),
        ).reset_index()
        _show_table(
            summary, width='stretch', hide_index=True,
            column_config={
                'observed_deposits': st.column_config.NumberColumn(format='$%.2f'),
                'observed_wagers': st.column_config.NumberColumn(format='$%.2f'),
            },
        )
        channel_chart = px.bar(
            summary, x='channel', y='observed_events', color='channel',
            color_discrete_map=CHANNEL_COLORS,
        )
        channel_chart.update_layout(
            height=220, showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='Observed events',
        )
        st.plotly_chart(channel_chart, width='stretch')

    st.subheader(f'Recent priority alerts | review budget {budget}')
    recent_alerts = queue[queue.priority_rank.le(budget)].sort_values(
        ['latest_activity', 'priority_rank'], ascending=[False, True]
    ).head(10)
    _show_table(
        recent_alerts[[
            'priority_rank', 'display_id', 'latest_activity', 'aml_fraud_score', 'safer_gambling_score',
            'primary_alert_reason', 'risk_band', 'investigation_status',
        ]],
        width='stretch', hide_index=True,
        column_config={
            'aml_fraud_score': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f'),
            'safer_gambling_score': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f'),
        },
    )
    st.caption('Risk indices prioritize human review and are not calibrated probabilities.')


def _investigation_queue(queue: pd.DataFrame, budget: int):
    _page_header(
        'TRIAGE', 'Investigation Queue',
        'Search and filter the ranked review workload without changing model outputs.',
    )
    one, two, three, four = st.columns([1.5, 1, 1, 1])
    search = one.text_input('Search player or entity', placeholder='P000123 or source account')
    bands = two.multiselect(
        'Risk band', ['critical', 'high', 'moderate', 'low'],
        default=['critical', 'high', 'moderate'],
    )
    qualities = three.multiselect(
        'Data quality', sorted(queue.data_quality.dropna().unique()),
        default=list(sorted(queue.data_quality.dropna().unique())),
    )
    statuses = four.multiselect(
        'Status', ['New', 'Under Review', 'Escalated', 'Closed'],
        default=['New', 'Under Review', 'Escalated'],
    )
    within_budget = st.toggle('Show only accounts inside the current review budget', value=True)
    filtered = queue.copy()
    if search:
        filtered = filtered[filtered.display_id.astype(str).str.contains(search, case=False, regex=False)]
    if bands:
        filtered = filtered[filtered.risk_band.isin(bands)]
    if qualities:
        filtered = filtered[filtered.data_quality.isin(qualities)]
    if statuses:
        filtered = filtered[filtered.investigation_status.isin(statuses)]
    if within_budget:
        filtered = filtered[filtered.priority_rank.le(budget)]

    st.markdown(
        f'<span class="sx-pill">{len(filtered):,} visible</span>'
        f'<span class="sx-pill">{budget} review capacity</span>',
        unsafe_allow_html=True,
    )
    display = filtered[[
        'priority_rank', 'display_id', 'aml_fraud_score', 'safer_gambling_score',
        'primary_risk_domain', 'primary_alert_reason', 'channel_coverage', 'data_quality',
        'identity_status', 'risk_band', 'investigation_status',
    ]]
    _show_table(
        display, width='stretch', hide_index=True, height=610,
        column_config={
            'aml_fraud_score': st.column_config.ProgressColumn('AML risk', min_value=0, max_value=100, format='%.1f'),
            'safer_gambling_score': st.column_config.ProgressColumn('Safer gambling', min_value=0, max_value=100, format='%.1f'),
            'channel_coverage': st.column_config.ProgressColumn('Channel coverage', min_value=0, max_value=1, format='%.0%%'),
        },
    )


def _player_360(data: DashboardData, queue: pd.DataFrame):
    _page_header(
        'CASE WORKSPACE', 'Player 360',
        'Unified history, behavioral context, evidence, and human investigation workflow.',
    )
    eligible = queue[queue.player_id.notna()].copy()
    player = st.selectbox(
        'Select fictional player',
        eligible.player_id.astype(str).tolist(),
        format_func=lambda pid: (
            f'{pid} | rank {int(eligible.loc[eligible.player_id.eq(pid), "priority_rank"].iloc[0])}'
        ),
    )
    row = eligible[eligible.player_id.eq(player)].iloc[0]
    profile = data.synthetic.players[data.synthetic.players.player_id.eq(player)].iloc[0]
    entity = row.activity_entity_id
    events = data.events[data.events.activity_entity_id.eq(entity)].copy()
    observed = events[events.observed.fillna(False)].copy()
    history = data.features[data.features.activity_entity_id.eq(entity)].copy()
    model = data.model_outputs[data.model_outputs.activity_entity_id.eq(entity)]

    a, b, c, d = st.columns(4)
    a.metric('AML / Fraud', f'{row.aml_fraud_score:.1f}')
    b.metric('Safer Gambling', f'{row.safer_gambling_score:.1f}')
    c.metric('Priority rank', f'#{int(row.priority_rank)}')
    d.metric('Identity confidence', 'Unknown' if pd.isna(row.identity_confidence) else f'{row.identity_confidence:.0%}')

    st.markdown('### Player profile')
    profile_columns = st.columns(5)
    for column, label, value in zip(
        profile_columns,
        ['Age band', 'Region', 'Account opened', 'Account status', 'Gaming preference'],
        [profile.age_band, profile.region, profile.account_open_date, profile.account_status, profile.gaming_preferences],
    ):
        column.markdown(f'<div class="sx-card"><small>{label}</small><br><b>{value}</b></div>', unsafe_allow_html=True)
    st.caption('Age and region are shown for context and fairness audit only; they are not risk-scoring inputs.')

    left, right = st.columns([1.15, .85])
    with left:
        st.subheader('Deposits by channel')
        weekly = observed.assign(week=observed.timestamp.dt.to_period('W').dt.start_time).groupby(
            ['week', 'channel'], as_index=False
        ).deposit_amount.sum()
        deposit_chart = px.line(
            weekly, x='week', y='deposit_amount', color='channel', markers=True,
            color_discrete_map=CHANNEL_COLORS,
        )
        deposit_chart.update_layout(
            height=340, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='Deposits (CAD)', legend_title_text='',
        )
        st.plotly_chart(deposit_chart, width='stretch')
    with right:
        st.subheader('Historical behavioral baseline')
        baseline = history[['timestamp', 'unified_deposit_amount_7d', 'unified_historical_median_spending']].copy()
        baseline = baseline.drop_duplicates('timestamp').set_index('timestamp').resample('7D').last().reset_index()
        baseline = baseline.melt('timestamp', var_name='measure', value_name='amount')
        baseline_chart = px.line(
            baseline, x='timestamp', y='amount', color='measure',
            color_discrete_map={
                'unified_deposit_amount_7d': '#22d3ee',
                'unified_historical_median_spending': '#a78bfa',
            },
        )
        baseline_chart.update_layout(
            height=340, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='Observed amount (CAD)', legend_title_text='',
        )
        st.plotly_chart(baseline_chart, width='stretch')

    st.subheader('Unified activity timeline')
    timeline = observed.copy()
    timeline['plot_amount'] = timeline.amount.fillna(0)
    timeline_chart = px.scatter(
        timeline, x='timestamp', y='plot_amount', color='channel', symbol='event_type',
        hover_data=['event_id', 'event_type', 'net_loss'], color_discrete_map=CHANNEL_COLORS,
    )
    timeline_chart.update_layout(
        height=330, margin=dict(l=5, r=5, t=10, b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
        font_color='#e6f0ff', xaxis_title='', yaxis_title='Observed event amount (CAD)',
    )
    st.plotly_chart(timeline_chart, width='stretch')

    peer_left, risk_right = st.columns([.75, 1.25])
    with peer_left:
        st.subheader('Peer comparison')
        latest = history.sort_values(['timestamp', 'event_id']).iloc[-1]
        peer_values = pd.DataFrame({
            'measure': ['Player 30-day deposits', 'Behavioral peer median'],
            'amount': [latest.unified_deposit_amount_30d, latest.peer_deposit_median_30d],
        })
        peer_chart = px.bar(peer_values, x='measure', y='amount', color='measure')
        peer_chart.update_layout(
            height=300, showlegend=False, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='CAD',
        )
        st.plotly_chart(peer_chart, width='stretch')
        if len(model):
            st.metric('Peer deviation index', f'{model.iloc[0].peer_deviation_score:.1f}')
    with risk_right:
        st.subheader('Risk score breakdown')
        ledger = _risk_ledger_frame(row)
        available = ledger[ledger.available].sort_values('score_contribution', ascending=True)
        breakdown = px.bar(
            available, x='score_contribution', y='signal', color='risk_domain', orientation='h',
            color_discrete_map={'AML / Fraud': '#2f81f7', 'Safer Gambling': '#22d3ee'},
        )
        breakdown.update_layout(
            height=360, margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='Exact score contribution', yaxis_title='',
            legend_title_text='',
        )
        st.plotly_chart(breakdown, width='stretch')

    case_records = st.session_state.get(STORE_KEY, {})
    status = case_records.get(f'CASE-{entity}', {}).get('status', 'New')
    case = build_investigation_case(row, events, feature_history=history, status=status)
    st.markdown('### Investigator copilot')
    for statement in case.why_flagged:
        st.markdown(escape_currency_markdown(statement))
    if case.historical_changes:
        st.markdown('**What changed**')
        for statement in case.historical_changes:
            st.markdown('- ' + escape_currency_markdown(statement))
    st.markdown(
        '**Evidence channels:** '
        + (', '.join(channel.replace('_', ' ').title() for channel in case.contributing_channels)
           if case.contributing_channels else 'Not established')
    )
    concern, alternative = st.columns(2)
    with concern:
        st.markdown('#### EVIDENCE FOR CONCERN')
        for statement in case.evidence_for_concern:
            st.markdown('- ' + escape_currency_markdown(statement))
    with alternative:
        st.markdown('#### EVIDENCE AGAINST OR ALTERNATIVE EXPLANATIONS')
        for statement in case.evidence_against_or_alternatives:
            st.markdown('- ' + escape_currency_markdown(statement))
    with st.expander('Supporting transactions and missing context', expanded=True):
        if case.supporting_transactions:
            _show_table(pd.DataFrame(case.supporting_transactions), width='stretch', hide_index=True)
        for statement in case.missing_or_uncertain:
            st.warning(statement)
    st.markdown('#### Recommended human-review actions')
    for statement in case.recommended_next_steps:
        st.markdown('- ' + escape_currency_markdown(statement))
    render_investigator_controls(case)

    with st.expander('Channel-specific transaction history'):
        channel_filter = st.multiselect(
            'Channels', sorted(events.channel.unique()), default=list(sorted(events.channel.unique())),
            key='player-channel-filter',
        )
        transaction_columns = [
            'event_id', 'timestamp', 'channel', 'event_type', 'amount', 'deposit_amount',
            'withdrawal_amount', 'wager_amount', 'payout_amount', 'net_loss', 'observed',
        ]
        _show_table(
            events[events.channel.isin(channel_filter)][transaction_columns].sort_values(
                'timestamp', ascending=False
            ),
            width='stretch', hide_index=True,
        )


def _network_intelligence(data: DashboardData):
    _page_header(
        'RELATIONSHIP ANALYSIS', 'Network Intelligence',
        'Observed account connections are investigative leads - not determinations of fraud.',
    )
    stats = data.network.statistics
    columns = st.columns(5)
    columns[0].metric('Source accounts', f"{stats['nodes']:,}")
    columns[1].metric('Relationships', f"{stats['edges']:,}")
    columns[2].metric('Components', f"{stats['connected_components']:,}")
    columns[3].metric('Largest cluster', f"{stats['largest_component']:,}")
    columns[4].metric('Unusual clusters', f"{stats['potentially_unusual_components']:,}")
    st.plotly_chart(_network_figure(data.network), width='stretch')
    st.caption(
        f'The interactive layout shows {len(data.network.plotly_nodes):,} evidence-ranked accounts '
        f'of {len(data.network.nodes):,}; all accounts remain included in graph metrics and scoring.'
    )

    components = data.network.components[data.network.components.edge_count.gt(0)].copy()
    if components.empty:
        st.info('No connected account clusters were observed for this configuration.')
        return
    flagged = components[components.ring_status.eq('Ring flagged')]
    if len(flagged):
        st.success(f'{len(flagged)} multi-evidence relationship ring(s) flagged for human review.')
        _show_table(
            flagged[[
                'component_id', 'ring_status', 'component_size', 'channels',
                'shared_identifier_edges', 'correlated_activity_edges',
            ]], width='stretch', hide_index=True,
        )
    ordered_components = components.sort_values(
        ['potentially_unusual', 'component_size'], ascending=[False, False]
    )
    selected = st.selectbox(
        'Inspect connected component', ordered_components.component_id.tolist(),
        format_func=lambda value: (
            f'{components.loc[components.component_id.eq(value), "ring_status"].iloc[0]} | '
            f'Cluster {value} | {int(components.loc[components.component_id.eq(value), "component_size"].iloc[0])} accounts'
        ),
    )
    members = data.network.nodes[data.network.nodes.component_id.eq(selected)]
    member_ids = set(members.node_id)
    edges = data.network.edges[
        data.network.edges.source.isin(member_ids) & data.network.edges.target.isin(member_ids)
    ]
    name_lookup = data.network.nodes.set_index('node_id').display_name.to_dict()
    edges = edges.assign(
        source_name=edges.source.map(name_lookup), target_name=edges.target.map(name_lookup)
    )
    summary = components[components.component_id.eq(selected)].iloc[0]
    if summary.ring_status == 'Ring flagged':
        st.warning(summary.explanation)
    else:
        st.info(summary.explanation)
    left, right = st.columns([.8, 1.2])
    with left:
        st.markdown('#### Cluster accounts')
        _show_table(
            members[[
                'display_name', 'channel', 'identity_status', 'identity_confidence',
                'degree', 'shared_identifier_count', 'investigation_weighted_degree',
            ]], width='stretch', hide_index=True,
        )
    with right:
        st.markdown('#### Relationship explanations')
        if len(edges):
            _show_table(
                edges[[
                    'source_name', 'target_name', 'relationship_types', 'shared_device_count',
                    'shared_payment_instrument_count', 'correlated_activity_windows',
                    'investigation_strength', 'explanation',
                ]], width='stretch', hide_index=True,
            )
        else:
            st.info('This component has no displayed relationship edges.')
    st.caption('A shared device or payment method alone never establishes fraud.')


def _validation(data: DashboardData, queue: pd.DataFrame, budget: int, player_count: int, seed: int):
    _page_header(
        'MODEL ASSURANCE', 'Validation & Fairness',
        'Held synthetic labels are used here only after operational scoring.',
    )
    comparable = {
        name: evaluate_ranked_scores(ranking, data.synthetic.scenario_labels, budget)
        for name, ranking in detector_scores(data.synthetic.players, data.operational_events)
        if name in {'Simple Baseline Rules', 'Blue Detector V2'}
    }
    st.subheader('Primary detector vs baseline | same population and review budget')
    baseline_column, primary_column = st.columns(2)
    for column, name, label in (
        (baseline_column, 'Simple Baseline Rules', 'Baseline | simple rules'),
        (primary_column, 'Blue Detector V2', 'Primary | Blue Detector V2'),
    ):
        result = comparable[name]
        with column:
            st.markdown(f'#### {label}')
            one, two, three = st.columns(3)
            one.metric('Recall', f"{result['recall_at_k']:.1%}", help='Recall at the current review budget')
            two.metric('Precision', f"{result['precision_at_k']:.1%}", help='Precision at the current review budget')
            three.metric('FN', result['false_negatives'], help='False negatives')
    st.caption('These headline values use the same detector functions, population, and budget as Arena Level 0.')

    metrics = evaluate_at_budget(queue, data.synthetic.scenario_labels, budget)
    columns = st.columns(6)
    columns[0].metric('Recall', f"{metrics['recall_at_k']:.1%}", help='Recall at the current review budget')
    columns[1].metric('Precision', f"{metrics['precision_at_k']:.1%}", help='Precision at the current review budget')
    columns[2].metric('TP', metrics['true_positives'], help='True positives')
    columns[3].metric('FP', metrics['false_positives'], help='False positives')
    columns[4].metric('FN', metrics['false_negatives'], help='False negatives')
    columns[5].metric('Unlinked', metrics['unlinked_alerts_not_labelled'])
    st.caption(
        'Metrics describe this configured synthetic run only. Unlinked alerts have no player-level held label '
        'and are reported separately rather than counted as false positives.'
    )

    scenario, channel = st.columns(2)
    with scenario:
        st.subheader('Performance by injected scenario')
        scenario_table = performance_by_scenario(queue, data.synthetic.scenario_labels, budget)
        _show_table(
            scenario_table, width='stretch', hide_index=True,
            column_config={
                'recall_at_k': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.1%%'),
            },
        )
    with channel:
        st.subheader('Performance by observed channel')
        channel_table = performance_by_channel(
            queue, data.synthetic.scenario_labels, data.events, budget
        )
        _show_table(
            channel_table, width='stretch', hide_index=True,
            column_config={
                'precision_at_k': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.1%%'),
                'recall_at_k': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.1%%'),
            },
        )

    st.subheader('Fairness audit')
    field = st.radio('Audit grouping', ['age_band', 'region'], horizontal=True)
    audit = fairness_audit(queue, data.synthetic.players, field, budget)
    chart, table = st.columns([1.1, .9])
    with chart:
        audit_chart = px.bar(
            audit, x=field, y='review_rate', color='review_rate',
            color_continuous_scale=['#17345a', '#2f81f7', '#22d3ee'],
        )
        audit_chart.update_layout(
            height=340, margin=dict(l=5, r=5, t=10, b=5), coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', yaxis_tickformat='.1%', yaxis_title='Review rate',
        )
        st.plotly_chart(audit_chart, width='stretch')
    with table:
        _show_table(
            audit, width='stretch', hide_index=True,
            column_config={
                'review_rate': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.1%%'),
            },
        )
    st.caption(
        'Age and region are audit attributes only and are not model inputs. Review rate partly reflects '
        'the deliberately stratified injected cases, not demographic scoring. Groups with n < 30 are '
        'marked low sample; equal review rates do not establish fairness.'
    )

    with st.expander('Cross-channel visibility stress test (legacy benchmark)'):
        st.write(
            'This preserves the earlier missing-observation stress test. It evaluates three transparent '
            'benchmark configurations and does not alter the current queue.'
        )
        if st.button('Run four visibility stress levels', type='primary'):
            results = []
            for level in range(4):
                altered = stress_test(
                    data.operational_events, data.synthetic.scenario_labels, level
                )
                result = evaluate_legacy_models(
                    data.synthetic.players, altered, data.synthetic.scenario_labels, budget
                )
                result['stress_level'] = level
                results.append(result)
            st.session_state['phase9_stress_results'] = pd.concat(results, ignore_index=True)
        if 'phase9_stress_results' in st.session_state:
            stress = st.session_state.phase9_stress_results
            stress_chart = px.line(
                stress, x='stress_level', y='recall_at_k', color='model', markers=True,
            )
            stress_chart.update_layout(
                height=330, margin=dict(l=5, r=5, t=10, b=5),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
                font_color='#e6f0ff', yaxis_tickformat='.1%',
            )
            st.plotly_chart(stress_chart, width='stretch')
            _show_table(stress, width='stretch', hide_index=True)


def _red_vs_blue_arena(player_count: int, seed: int, budget: int):
    _page_header(
        'DEFENSIVE STRESS LAB', 'Red Team vs Blue Team Arena',
        'Abstract fictional stressors test detection resilience without modelling real evasion tactics.',
    )
    st.warning(
        'Defensive synthetic evaluation only. Transformations are intentionally abstract and are not '
        'instructions for avoiding financial-crime or player-protection monitoring.'
    )
    left, right = st.columns([1.3, .7])
    experiment = left.selectbox(
        'Attack scenario selector',
        ['Behavioral Adaptation', 'Cross-Channel Visibility', 'Both Experiments'],
    )
    right.metric('Held-out alert budget', budget)
    if st.button('Run Experiment', type='primary'):
        st.session_state['phase10_arena_result'] = run_cached_arena(
            player_count, seed, budget
        )
    if 'phase10_arena_result' not in st.session_state:
        st.info(
            'Run the experiment to create separate development and held-out populations. '
            'No result is predetermined.'
        )
        return

    result = st.session_state.phase10_arena_result
    st.caption(
        f"Development seed {result.metadata['development_seed']} | Held-out seed "
        f"{result.metadata['evaluation_seed']} | {result.metadata['players_per_split']:,} players "
        f"per split | {result.metadata['events_per_split']:,} events per split"
    )
    show_behavior = experiment in {'Behavioral Adaptation', 'Both Experiments'}
    show_visibility = experiment in {'Cross-Channel Visibility', 'Both Experiments'}

    if show_behavior:
        st.markdown('### Experiment A | Behavioral adaptation')
        level = st.select_slider(
            'Inspect complexity level', options=list(BEHAVIOR_LEVELS), value=0,
            format_func=lambda value: f'Level {value}: {BEHAVIOR_LEVELS[value]}',
        )
        held = result.behavioral_metrics[
            result.behavioral_metrics.dataset_split.eq('held_out_evaluation')
        ].copy()
        selected = held[held.complexity_level.eq(level)]
        comparison = selected.melt(
            id_vars='detector', value_vars=['recall_at_k', 'precision_at_k'],
            var_name='metric', value_name='value',
        )
        detector_chart = px.bar(
            comparison, x='detector', y='value', color='metric', barmode='group',
            color_discrete_map={'recall_at_k': '#22d3ee', 'precision_at_k': '#a78bfa'},
        )
        detector_chart.update_layout(
            height=360, margin=dict(l=5, r=5, t=15, b=5), yaxis_tickformat='.0%',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='Held-out metric',
            legend_title_text='',
        )
        st.plotly_chart(detector_chart, width='stretch')

        trend, failures = st.columns([1.2, .8])
        with trend:
            st.markdown('#### Recall vs attack complexity')
            recall_chart = px.line(
                held, x='complexity_level', y='recall_at_k', color='detector', markers=True,
            )
            recall_chart.update_layout(
                height=340, margin=dict(l=5, r=5, t=10, b=5), yaxis_tickformat='.0%',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
                font_color='#e6f0ff', xaxis={'dtick': 1}, legend_title_text='',
            )
            st.plotly_chart(recall_chart, width='stretch')
        with failures:
            st.markdown('#### False-negative counts')
            failure_chart = px.bar(
                held, x='complexity_level', y='false_negatives', color='detector', barmode='group',
            )
            failure_chart.update_layout(
                height=340, margin=dict(l=5, r=5, t=10, b=5),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
                font_color='#e6f0ff', xaxis={'dtick': 1}, legend_title_text='',
            )
            st.plotly_chart(failure_chart, width='stretch')
        _show_table(
            selected[[
                'detector', 'review_workload', 'true_positives', 'false_positives',
                'false_negatives', 'precision_at_k', 'recall_at_k', 'reviews_per_detection',
            ]], width='stretch', hide_index=True,
        )

    if show_visibility:
        st.markdown('### Experiment B | Cross-channel visibility')
        held_visibility = result.visibility_metrics[
            result.visibility_metrics.dataset_split.eq('held_out_evaluation')
        ].copy()
        visibility_chart = px.bar(
            held_visibility, x='condition', y='recall_at_k', color='detector', barmode='group',
        )
        visibility_chart.update_layout(
            height=390, margin=dict(l=5, r=5, t=15, b=5), yaxis_tickformat='.0%',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,20,38,.35)',
            font_color='#e6f0ff', xaxis_title='', yaxis_title='Held-out Recall@K',
            legend_title_text='',
        )
        st.plotly_chart(visibility_chart, width='stretch')
        _show_table(
            held_visibility[[
                'condition', 'detector', 'review_workload', 'true_positives',
                'false_positives', 'false_negatives', 'precision_at_k', 'recall_at_k',
            ]], width='stretch', hide_index=True,
        )
        coverage = result.coverage_recall[
            result.coverage_recall.dataset_split.eq('held_out_evaluation')
            & result.coverage_recall.experiment.eq('cross_channel_visibility')
        ]
        st.markdown('#### Recall by observed channel coverage')
        _show_table(coverage, width='stretch', hide_index=True)

    held_all = pd.concat([
        result.behavioral_metrics[result.behavioral_metrics.dataset_split.eq('held_out_evaluation')],
        result.visibility_metrics[result.visibility_metrics.dataset_split.eq('held_out_evaluation')],
    ], ignore_index=True)
    strongest = held_all.sort_values(
        ['true_positives', 'recall_at_k'], ascending=False
    ).iloc[0]
    weakest = held_all.sort_values(
        ['false_negatives', 'recall_at_k'], ascending=[False, True]
    ).iloc[0]
    st.markdown('### Result interpretation')
    success, failure = st.columns(2)
    success.success(
        f"Successful detections: {strongest.detector} detected {int(strongest.true_positives)} "
        f"of {int(strongest.positives)} held-out positives under {strongest.condition}."
    )
    failure.error(
        f"Model failures: {weakest.detector} missed {int(weakest.false_negatives)} "
        f"held-out positives under {weakest.condition}."
    )
    st.info(
        'Differences are measured outcomes from this synthetic run, not guaranteed improvements. '
        'Development results are retained separately and the held-out population is not used to tune detectors.'
    )
    with st.expander('Held-out recall by injected scenario'):
        scenarios = result.scenario_recall[
            result.scenario_recall.dataset_split.eq('held_out_evaluation')
        ]
        _show_table(scenarios, width='stretch', hide_index=True)


_inject_theme()

with st.sidebar:
    st.markdown('## Sentinel X')
    st.caption('FICTIONAL SYNTHETIC DATA | LOCAL PROTOTYPE')
    page = st.radio(
        'Workspace',
        ['Overview', 'Investigation Queue', 'Player 360', 'Network Intelligence',
         'Validation & Fairness', 'Red vs Blue Arena'],
    )
    st.divider()
    st.markdown('### Environment')
    player_count = st.select_slider(
        'Synthetic player count', options=[100, 250, 500, 1_000, 2_000], value=DEFAULT_UI_PLAYERS
    )
    seed = int(st.number_input('Random seed', value=RANDOM_SEED, min_value=0))
    cross_channel = st.toggle('Cross-channel integration', value=True)
    budget = max(1, int(round(player_count * .05)))
    st.metric('Review budget (5%)', budget)

data = prepare_dashboard(player_count, seed, cross_channel)
case_records = st.session_state.get(STORE_KEY, {})
queue = ranked_queue(data.scored, budget, case_records)

with st.sidebar:
    st.divider()
    st.markdown('### Detection model')
    st.code(data.model_version, language=None)
    st.caption(
        f'{len(data.events):,} normalized records | '
        f'{"3-channel" if cross_channel else "online-casino only"} mode'
    )
    st.info('Risk scores are review indices - not fraud or harm probabilities.')

if page == 'Overview':
    _overview(data, queue, budget)
elif page == 'Investigation Queue':
    _investigation_queue(queue, budget)
elif page == 'Player 360':
    _player_360(data, queue)
elif page == 'Network Intelligence':
    _network_intelligence(data)
elif page == 'Validation & Fairness':
    _validation(data, queue, budget, player_count, seed)
else:
    _red_vs_blue_arena(player_count, seed, budget)

st.divider()
st.caption(
    'Sentinel X is a fictional hackathon prototype. It does not represent OLG internal architecture and '
    'does not take automated action against players or accounts.'
)
