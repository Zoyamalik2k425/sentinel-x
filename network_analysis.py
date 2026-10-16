"""Account relationship intelligence for fictional Sentinel X data.

The graph uses observed source-account evidence only. It never consumes final
risk scores or alert decisions, and every relationship is an investigation
lead rather than a determination of fraud.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


FORBIDDEN_SCORING_INPUTS = frozenset({
    'priority', 'alert', 'anomaly_severity', 'behavioral_change_score',
    'peer_deviation_score', 'model_priority_index', 'aml_fraud_score',
    'safer_gambling_score', 'ground_truth_label', 'scenario_type',
})
REQUIRED_EVENT_FIELDS = frozenset({
    'source_account_id', 'channel', 'timestamp', 'event_id', 'observed',
    'canonical_player_id', 'proposed_canonical_player_id', 'identity_status',
    'identity_confidence', 'device_id', 'payment_instrument_id',
})


@dataclass(frozen=True)
class NetworkAnalysisResult:
    graph: nx.Graph
    nodes: pd.DataFrame
    edges: pd.DataFrame
    components: pd.DataFrame
    plotly_nodes: pd.DataFrame
    plotly_edges: pd.DataFrame

    @property
    def statistics(self) -> dict[str, int | float]:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'connected_components': nx.number_connected_components(self.graph),
            'isolated_accounts': nx.number_of_isolates(self.graph),
            'largest_component': int(self.components.component_size.max()) if len(self.components) else 0,
            'potentially_unusual_components': int(self.components.potentially_unusual.sum())
            if len(self.components) else 0,
        }


def _account_metadata(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values(['timestamp', 'event_id'])
    return ordered.groupby(['channel', 'source_account_id'], dropna=False).agg(
        canonical_player_id=('canonical_player_id', 'last'),
        proposed_canonical_player_id=('proposed_canonical_player_id', 'last'),
        identity_status=('identity_status', 'last'),
        identity_confidence=('identity_confidence', 'min'),
        first_observed=('timestamp', 'min'),
        last_observed=('timestamp', 'max'),
        observed_events=('observed', 'sum'),
        source_records=('event_id', 'count'),
    ).reset_index()


def _node_id(channel: str, account: str) -> str:
    return f'{channel}:{account}'


def _new_evidence():
    return {
        'relationship_types': set(),
        'shared_devices': set(),
        'shared_payment_instruments': set(),
        'shared_device_group_sizes': [],
        'shared_payment_group_sizes': [],
        'correlated_activity_windows': 0,
        'confirmed_ownership': False,
    }


def _pair(left, right):
    return tuple(sorted((left, right)))


def _add_identifier_relationships(
    observed: pd.DataFrame,
    account_lookup: pd.Series,
    field: str,
    relationship_type: str,
    accumulator: dict,
    maximum_group_size: int,
):
    usable = observed[observed[field].notna()].copy()
    for identifier, group in usable.groupby(field):
        accounts = sorted(set(account_lookup.loc[group.index]))
        if len(accounts) < 2 or len(accounts) > maximum_group_size:
            continue
        for left, right in combinations(accounts, 2):
            evidence = accumulator[_pair(left, right)]
            evidence['relationship_types'].add(relationship_type)
            if field == 'device_id':
                evidence['shared_devices'].add(str(identifier))
                evidence['shared_device_group_sizes'].append(len(accounts))
            else:
                evidence['shared_payment_instruments'].add(str(identifier))
                evidence['shared_payment_group_sizes'].append(len(accounts))


def _add_correlated_activity(
    observed: pd.DataFrame,
    account_lookup: pd.Series,
    accumulator: dict,
    activity_window: str,
    minimum_correlated_windows: int,
    maximum_accounts_per_window: int,
):
    frame = observed.copy()
    frame['_activity_window'] = frame.timestamp.dt.floor(activity_window)
    overlap_counts = defaultdict(int)
    for _, group in frame.groupby('_activity_window'):
        accounts = sorted(set(account_lookup.loc[group.index]))
        if len(accounts) < 2 or len(accounts) > maximum_accounts_per_window:
            continue
        for left, right in combinations(accounts, 2):
            overlap_counts[_pair(left, right)] += 1
    for pair, count in overlap_counts.items():
        if count >= minimum_correlated_windows:
            evidence = accumulator[pair]
            evidence['relationship_types'].add('correlated_activity')
            evidence['correlated_activity_windows'] = count


def _add_confirmed_ownership(metadata: pd.DataFrame, accumulator: dict):
    confirmed = metadata[
        metadata.identity_status.eq('confirmed') & metadata.canonical_player_id.notna()
    ].copy()
    confirmed['node_id'] = [
        _node_id(row.channel, row.source_account_id) for row in confirmed.itertuples()
    ]
    for _, group in confirmed.groupby('canonical_player_id'):
        records = group[['node_id', 'channel']].drop_duplicates()
        for left, right in combinations(records.itertuples(index=False), 2):
            if left.channel == right.channel:
                continue
            evidence = accumulator[_pair(left.node_id, right.node_id)]
            evidence['relationship_types'].add('confirmed_cross_channel_ownership')
            evidence['confirmed_ownership'] = True


def _edge_strength(evidence: dict) -> tuple[float, float]:
    device = min(.45, .20 * len(evidence['shared_devices']))
    payment = min(.55, .25 * len(evidence['shared_payment_instruments']))
    correlation = min(.70, .12 * evidence['correlated_activity_windows'])
    ownership = .90 if evidence['confirmed_ownership'] else 0.0
    relationship = 1 - np.prod([1 - device, 1 - payment, 1 - correlation, 1 - ownership])
    investigation = 1 - np.prod([1 - device, 1 - payment, 1 - correlation])
    return float(relationship), float(investigation)


def explain_relationship(left: str, right: str, evidence: dict) -> str:
    """Create a plain-language explanation without asserting wrongdoing."""
    statements = []
    device_count = len(evidence['shared_devices'])
    payment_count = len(evidence['shared_payment_instruments'])
    if device_count:
        accounts = max(evidence['shared_device_group_sizes'], default=2)
        statements.append(
            f'{accounts} accounts share {device_count} synthetic device identifier'
            f'{"s" if device_count != 1 else ""}'
        )
    if payment_count:
        accounts = max(evidence['shared_payment_group_sizes'], default=2)
        statements.append(
            f'{accounts} accounts share {payment_count} synthetic payment instrument token'
            f'{"s" if payment_count != 1 else ""}'
        )
    windows = evidence['correlated_activity_windows']
    if windows:
        statements.append(f'the accounts exhibit activity in {windows} overlapping time windows')
    if evidence['confirmed_ownership']:
        statements.append('the source accounts have confirmed synthetic cross-channel ownership')
    if not statements:
        statements.append('an observed relationship is present')
    message = '; '.join(statements)
    return message[0].upper() + message[1:] + '. This is an investigation lead, not proof of fraud.'


def _component_table(graph: nx.Graph) -> tuple[pd.DataFrame, dict[str, int]]:
    rows, memberships = [], {}
    components = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
    for component_id, members in enumerate(components):
        subgraph = graph.subgraph(members)
        edges = list(subgraph.edges(data=True))
        correlated = sum('correlated_activity' in data['relationship_types'] for _, _, data in edges)
        identifier_edges = sum(
            bool({'shared_device', 'shared_payment_instrument'}.intersection(data['relationship_types']))
            for _, _, data in edges
        )
        investigation_strengths = [data['investigation_strength'] for _, _, data in edges]
        strong_multi_identifier = (
            identifier_edges >= 3
            and sum(data['shared_device_count'] > 0 for _, _, data in edges) >= 2
            and sum(data['shared_payment_instrument_count'] > 0 for _, _, data in edges) >= 2
        )
        unusual = len(members) >= 3 and identifier_edges >= 2 and (
            correlated >= 2 or strong_multi_identifier
        )
        for node in members:
            memberships[node] = component_id
        rows.append({
            'component_id': component_id,
            'component_size': len(members),
            'edge_count': subgraph.number_of_edges(),
            'channels': '|'.join(sorted(set(graph.nodes[node]['channel'] for node in members))),
            'shared_identifier_edges': identifier_edges,
            'correlated_activity_edges': correlated,
            'confirmed_ownership_edges': sum(
                'confirmed_cross_channel_ownership' in data['relationship_types'] for _, _, data in edges
            ),
            'average_relationship_strength': float(np.mean([
                data['relationship_strength'] for _, _, data in edges
            ])) if edges else 0.0,
            'potentially_unusual': unusual,
            'ring_status': 'Ring flagged' if unusual else 'No ring flag',
            'explanation': (
                f'Ring flagged: {len(members)} accounts form a cluster with {identifier_edges} shared-identifier '
                f'and {correlated} repeated-activity relationships. Human review is required.'
                if unusual else
                'No multi-evidence unusual-cluster condition was met.'
            ),
        })
    return pd.DataFrame(rows), memberships


def _plotly_frames(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
    maximum_nodes: int | None = None,
):
    plot_nodes = nodes.copy()
    if maximum_nodes is not None and len(plot_nodes) > maximum_nodes:
        # Layout cost grows quickly for thousands of accounts. Prefer nodes
        # with investigative evidence, then other connected accounts. This
        # affects only visualization; graph metrics still use every node.
        ranked_nodes = plot_nodes.assign(
            _connected=plot_nodes.degree.gt(0).astype(int),
        ).sort_values(
            ['potentially_unusual_cluster', 'investigation_weighted_degree',
             '_connected', 'weighted_degree', 'node_id'],
            ascending=[False, False, False, False, True],
        )
        included_order = []
        ranked_edges = edges.sort_values(
            ['investigation_strength', 'relationship_strength'],
            ascending=[False, False],
        )
        included = set()
        for edge in ranked_edges.itertuples():
            additions = [node for node in (edge.source, edge.target) if node not in included]
            if len(included) + len(additions) <= maximum_nodes:
                included.update(additions)
                included_order.extend(additions)
            if len(included) >= maximum_nodes:
                break
        if len(included) < maximum_nodes:
            additions = ranked_nodes.loc[
                ~ranked_nodes.node_id.isin(included), 'node_id'
            ].head(maximum_nodes - len(included)).tolist()
            included.update(additions)
            included_order.extend(additions)
        plot_nodes = ranked_nodes.set_index('node_id').loc[included_order].reset_index().drop(
            columns='_connected'
        )
        plot_graph = graph.subgraph(included).copy()
        plot_edges = edges[edges.source.isin(included) & edges.target.isin(included)].copy()
    else:
        plot_graph = graph
        plot_edges = edges.copy()
    positions = nx.spring_layout(plot_graph, seed=seed, weight='relationship_strength')
    plot_nodes['x'] = plot_nodes.node_id.map(lambda node: float(positions[node][0]))
    plot_nodes['y'] = plot_nodes.node_id.map(lambda node: float(positions[node][1]))
    plot_nodes['hover_text'] = plot_nodes.apply(
        lambda row: (
            f"{row.display_name}<br>Channel: {row.channel}<br>Identity: {row.identity_status}"
            f"<br>Degree: {row.degree}<br>Component size: {row.component_size}"
        ), axis=1,
    )
    if len(plot_edges):
        plot_edges['x0'] = plot_edges.source.map(lambda node: float(positions[node][0]))
        plot_edges['y0'] = plot_edges.source.map(lambda node: float(positions[node][1]))
        plot_edges['x1'] = plot_edges.target.map(lambda node: float(positions[node][0]))
        plot_edges['y1'] = plot_edges.target.map(lambda node: float(positions[node][1]))
        plot_edges['hover_text'] = plot_edges.explanation
    return plot_nodes, plot_edges


def build_relationship_graph(
    events: pd.DataFrame,
    activity_window: str = '1h',
    minimum_correlated_windows: int = 3,
    maximum_identifier_group_size: int = 20,
    maximum_accounts_per_window: int = 20,
    layout_seed: int = 42,
    maximum_plot_nodes: int | None = None,
) -> NetworkAnalysisResult:
    """Build account graph, metrics, explanations, and Plotly-ready coordinates."""
    forbidden = FORBIDDEN_SCORING_INPUTS.intersection(events.columns)
    if forbidden:
        raise ValueError(f'Final scores/labels cannot enter network analysis: {sorted(forbidden)}')
    missing = REQUIRED_EVENT_FIELDS.difference(events.columns)
    if missing:
        raise ValueError(f'Missing network event fields: {sorted(missing)}')
    frame = events.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    metadata = _account_metadata(frame)
    metadata['node_id'] = [
        _node_id(row.channel, row.source_account_id) for row in metadata.itertuples()
    ]
    lookup_frame = frame[['channel', 'source_account_id']].copy()
    account_lookup = pd.Series(
        [_node_id(row.channel, row.source_account_id) for row in lookup_frame.itertuples()],
        index=frame.index,
    )
    observed = frame[frame.observed.fillna(False)].copy()
    evidence = defaultdict(_new_evidence)
    _add_identifier_relationships(
        observed, account_lookup, 'device_id', 'shared_device', evidence,
        maximum_identifier_group_size,
    )
    _add_identifier_relationships(
        observed, account_lookup, 'payment_instrument_id', 'shared_payment_instrument', evidence,
        maximum_identifier_group_size,
    )
    _add_correlated_activity(
        observed, account_lookup, evidence, activity_window, minimum_correlated_windows,
        maximum_accounts_per_window,
    )
    _add_confirmed_ownership(metadata, evidence)

    graph = nx.Graph()
    for row in metadata.itertuples(index=False):
        graph.add_node(row.node_id, **row._asdict())
    edge_rows = []
    for (left, right), values in evidence.items():
        strength, investigation_strength = _edge_strength(values)
        attributes = {
            'relationship_types': tuple(sorted(values['relationship_types'])),
            'relationship_strength': strength,
            'investigation_strength': investigation_strength,
            'shared_device_count': len(values['shared_devices']),
            'shared_payment_instrument_count': len(values['shared_payment_instruments']),
            'correlated_activity_windows': values['correlated_activity_windows'],
            'confirmed_ownership': values['confirmed_ownership'],
            'explanation': explain_relationship(left, right, values),
        }
        graph.add_edge(left, right, **attributes)
        edge_rows.append({'source': left, 'target': right, **attributes})

    components, memberships = _component_table(graph)
    node_rows = []
    for node, attributes in graph.nodes(data=True):
        component_id = memberships[node]
        component = components.loc[components.component_id == component_id].iloc[0]
        incident = list(graph.edges(node, data=True))
        identity_status = attributes.get('identity_status', 'unlinked')
        canonical = attributes.get('canonical_player_id')
        channel_label = str(attributes.get('channel', 'source')).replace('_', ' ').title()
        display_name = (
            f'Player {canonical} · {channel_label}'
            if identity_status == 'confirmed' and pd.notna(canonical)
            else f'{str(identity_status).replace("_", " ").title()} · {channel_label} account '
                 f'{attributes.get("source_account_id", node)}'
        )
        node_rows.append({
            'node_id': node,
            'display_name': display_name,
            **attributes,
            'component_id': component_id,
            'component_size': int(component.component_size),
            'degree': int(graph.degree(node)),
            'weighted_degree': float(graph.degree(node, weight='relationship_strength')),
            # Confirmed ownership is useful for identity resolution but carries
            # zero investigation strength.  Keeping the two degree measures
            # separate prevents ordinary cross-channel ownership from becoming
            # an AML signal downstream.
            'investigation_weighted_degree': float(
                graph.degree(node, weight='investigation_strength')
            ),
            'shared_identifier_count': int(sum(
                data['shared_device_count'] + data['shared_payment_instrument_count']
                for _, _, data in incident
            )),
            'maximum_relationship_strength': float(max(
                (data['relationship_strength'] for _, _, data in incident), default=0.0
            )),
            'maximum_investigation_strength': float(max(
                (data['investigation_strength'] for _, _, data in incident), default=0.0
            )),
            'potentially_unusual_cluster': bool(component.potentially_unusual),
        })
    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows)
    if edges.empty:
        edges = pd.DataFrame(columns=[
            'source', 'target', 'relationship_types', 'relationship_strength',
            'investigation_strength', 'shared_device_count',
            'shared_payment_instrument_count', 'correlated_activity_windows',
            'confirmed_ownership', 'explanation',
        ])
    if maximum_plot_nodes is not None and maximum_plot_nodes < 1:
        raise ValueError('maximum_plot_nodes must be positive or None')
    plotly_nodes, plotly_edges = _plotly_frames(
        graph, nodes, edges, layout_seed, maximum_plot_nodes
    )
    return NetworkAnalysisResult(graph, nodes, edges, components, plotly_nodes, plotly_edges)
