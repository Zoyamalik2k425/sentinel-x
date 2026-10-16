"""Leakage-safe validation for the Sentinel X decision-support prototype.

Synthetic scenario labels are consumed only after operational scores have
been produced. They are never joined into detector features or model fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from data_gen import generate

from scoring import score
from signals import GROUND_TRUTH_COLUMNS, compute


def evaluate_ranked_scores(
    ranked_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    budget: int,
) -> dict[str, float | int]:
    """Evaluate a ``player_id``/``score`` frame at a fixed review capacity."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(ranked_scores.columns)
    if leaked:
        raise ValueError(f'Ground-truth columns cannot enter detector scores: {sorted(leaked)}')
    required = {'player_id', 'score'}
    missing = required.difference(ranked_scores.columns)
    if missing:
        raise ValueError(f'Missing ranked score columns: {sorted(missing)}')
    if budget < 1:
        raise ValueError('budget must be positive')
    population = ranked_scores.drop_duplicates('player_id').copy()
    reviewed_rows = population.sort_values(
        ['score', 'player_id'], ascending=[False, True], kind='stable'
    ).head(min(budget, len(population)))
    reviewed = set(reviewed_rows.player_id.astype(str))
    positive = set(
        ground_truth.loc[ground_truth.ground_truth_label.astype(bool), 'player_id'].astype(str)
    )
    true_positive = reviewed & positive
    false_positive = reviewed - positive
    false_negative = positive - reviewed
    return {
        'population': len(population),
        'alert_budget': min(budget, len(population)),
        'review_workload': len(reviewed),
        'positives': len(positive),
        'true_positives': len(true_positive),
        'false_positives': len(false_positive),
        'false_negatives': len(false_negative),
        'precision_at_k': len(true_positive) / max(len(reviewed), 1),
        'recall_at_k': len(true_positive) / max(len(positive), 1),
        'reviews_per_detection': len(reviewed) / max(len(true_positive), 1),
    }


def reviewed_player_ids(ranked_scores: pd.DataFrame, budget: int) -> set[str]:
    return set(
        ranked_scores.drop_duplicates('player_id').sort_values(
            ['score', 'player_id'], ascending=[False, True], kind='stable'
        ).head(min(budget, len(ranked_scores))).player_id.astype(str)
    )


def recall_by_scenario(
    ranked_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    reviewed = reviewed_player_ids(ranked_scores, budget)
    labels = ground_truth.copy()
    labels['reviewed'] = labels.player_id.astype(str).isin(reviewed)
    rows = []
    for scenario, group in labels.groupby('scenario_type', sort=True):
        positive = group.ground_truth_label.astype(bool)
        detected = group.reviewed & positive
        rows.append({
            'scenario_type': scenario,
            'positive_players': int(positive.sum()),
            'detected_players': int(detected.sum()),
            'false_negatives': int((positive & ~group.reviewed).sum()),
            'recall_at_k': int(detected.sum()) / max(int(positive.sum()), 1)
            if positive.any() else np.nan,
        })
    return pd.DataFrame(rows)


def player_channel_coverage(transactions: pd.DataFrame) -> pd.DataFrame:
    observed = transactions[transactions.observed.fillna(False)]
    coverage = observed.groupby('player_id').channel.nunique().rename('active_channels').reset_index()
    coverage['channel_coverage'] = coverage.active_channels / 3
    coverage['coverage_band'] = pd.cut(
        coverage.active_channels,
        bins=[-np.inf, 1, 2, np.inf],
        labels=['single_channel', 'partial_cross_channel', 'complete_cross_channel'],
    ).astype(str)
    return coverage


def recall_by_coverage(
    ranked_scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    transactions: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    reviewed = reviewed_player_ids(ranked_scores, budget)
    labels = ground_truth[ground_truth.ground_truth_label.astype(bool)].copy()
    coverage = player_channel_coverage(transactions)
    labels = labels.merge(coverage, on='player_id', how='left')
    labels['coverage_band'] = labels.coverage_band.fillna('no_observed_channel')
    labels['reviewed'] = labels.player_id.astype(str).isin(reviewed)
    return labels.groupby('coverage_band', dropna=False).agg(
        positive_players=('player_id', 'count'),
        detected_players=('reviewed', 'sum'),
        recall_at_k=('reviewed', 'mean'),
    ).reset_index()


def detector_scores(profiles, transactions):
    """Yield three label-free defensive detector rankings."""
    configurations = [
        ('Simple Baseline Rules', False, False),
        ('Blue Detector V1', True, False),
        ('Blue Detector V2', True, True),
    ]
    for name, cross_channel, hardened in configurations:
        features = compute(profiles, transactions, cross_channel, hardened)
        scored = score(features)
        yield name, scored[['player_id', 'priority']].rename(columns={'priority': 'score'})


def evaluate(profiles, tx, ground_truth, k=40):
    """Backward-compatible three-detector held-label evaluation."""
    rows = []
    display_names = {
        'Simple Baseline Rules': 'Single-channel V1',
        'Blue Detector V1': 'Cross-channel V1',
        'Blue Detector V2': 'Cross-channel V2',
    }
    for name, ranked in detector_scores(profiles, tx):
        metrics = evaluate_ranked_scores(ranked, ground_truth, k)
        rows.append({
            'model': display_names[name],
            'alerts': metrics['review_workload'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'precision_at_k': metrics['precision_at_k'],
            'recall_at_k': metrics['recall_at_k'],
        })
    return pd.DataFrame(rows)


def group_audit(scored, field='age_band', k=40):
    frame = scored.copy()
    frame['flagged'] = False
    frame.loc[frame.head(min(k, len(frame))).index, 'flagged'] = True
    return frame.groupby(field, dropna=False).agg(
        players=('player_id', 'count'),
        flags=('flagged', 'sum'),
        flag_rate=('flagged', 'mean'),
    ).reset_index()


RISK_DOMAINS = ('aml_fraud', 'safer_gambling')
DETECTOR_ORDER = (
    'Fixed Threshold Baseline',
    'Single-Channel Detection',
    'Cross-Channel Detection',
    'Cross-Channel + Network Intelligence',
)


@dataclass(frozen=True)
class ValidationResult:
    """Computed held-out assurance tables and the settings that produced them."""

    overall_metrics: pd.DataFrame
    scenario_recall: pd.DataFrame
    channel_recall: pd.DataFrame
    sensitivity: pd.DataFrame
    fairness_by_age: pd.DataFrame
    fairness_by_region: pd.DataFrame
    disparity_summary: pd.DataFrame
    metadata: dict[str, int | float | str]


def risk_detector_scores(profiles: pd.DataFrame, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce four comparable label-free score tables for the same population."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(transactions.columns) | GROUND_TRUTH_COLUMNS.intersection(profiles.columns)
    if leaked:
        raise ValueError(f'Ground-truth columns cannot enter detector inputs: {sorted(leaked)}')

    single_features = compute(profiles, transactions, cross_channel=False, hardened=False)
    single_scored = score(single_features)
    cross_features = compute(profiles, transactions, cross_channel=True, hardened=False)
    cross_scored = score(cross_features)
    network_features = compute(profiles, transactions, cross_channel=True, hardened=True)
    network_scored = score(network_features)

    baseline = single_features[['player_id', 'change_signal', 'loss_signal', 'geo_signal']].copy()
    baseline['aml_fraud_score'] = 100 * baseline[['change_signal', 'geo_signal']].max(axis=1)
    baseline['safer_gambling_score'] = 100 * baseline[['change_signal', 'loss_signal']].max(axis=1)
    baseline = baseline[['player_id', 'aml_fraud_score', 'safer_gambling_score']]

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[['player_id', 'aml_fraud_score', 'safer_gambling_score']].copy()

    return {
        DETECTOR_ORDER[0]: baseline,
        DETECTOR_ORDER[1]: select(single_scored),
        DETECTOR_ORDER[2]: select(cross_scored),
        DETECTOR_ORDER[3]: select(network_scored),
    }


def _domain_positive_players(labels: pd.DataFrame, domain: str) -> set[str]:
    if domain not in RISK_DOMAINS:
        raise ValueError(f'Unknown risk domain: {domain}')
    applies = labels.target_risk_category.isin([domain, 'both'])
    return set(labels.loc[labels.ground_truth_label.astype(bool) & applies, 'player_id'].astype(str))


def classification_metrics(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    domain: str,
    budget: int,
    threshold: float = 60.0,
) -> dict[str, float | int]:
    """Compute threshold and fixed-capacity metrics for one independent risk domain."""
    score_column = f'{domain}_score'
    required = {'player_id', score_column}
    if missing := required.difference(scores.columns):
        raise ValueError(f'Missing score columns: {sorted(missing)}')
    if GROUND_TRUTH_COLUMNS.intersection(scores.columns):
        raise ValueError('Ground-truth labels cannot be present in scored inputs')
    population = scores.drop_duplicates('player_id').copy()
    population['player_id'] = population.player_id.astype(str)
    positive = _domain_positive_players(labels, domain) & set(population.player_id)
    threshold_alerts = set(population.loc[population[score_column].ge(threshold), 'player_id'])
    ranked = population.sort_values([score_column, 'player_id'], ascending=[False, True], kind='stable')
    top_k = set(ranked.head(min(budget, len(ranked))).player_id)

    def counts(flagged: set[str]):
        tp = len(flagged & positive)
        fp = len(flagged - positive)
        fn = len(positive - flagged)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        return tp, fp, fn, precision, recall, f1

    tp, fp, fn, precision, recall, f1 = counts(threshold_alerts)
    top_tp, top_fp, top_fn, precision_k, recall_k, _ = counts(top_k)
    return {
        'population': len(population), 'positives': len(positive),
        'threshold': float(threshold), 'alert_volume': len(threshold_alerts),
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'alert_budget': min(budget, len(population)), 'review_workload': len(top_k),
        'top_k_true_positives': top_tp, 'top_k_false_positives': top_fp,
        'top_k_false_negatives': top_fn,
        'precision_at_k': precision_k, 'recall_at_k': recall_k,
    }


def _scenario_domain_recall(scores, labels, detector, domain, budget):
    column = f'{domain}_score'
    reviewed = set(scores.sort_values([column, 'player_id'], ascending=[False, True]).head(budget).player_id.astype(str))
    relevant = labels[
        labels.ground_truth_label.astype(bool)
        & labels.target_risk_category.isin([domain, 'both'])
    ]
    rows = []
    for scenario, group in relevant.groupby('scenario_type', sort=True):
        players = set(group.player_id.astype(str))
        detected = len(players & reviewed)
        rows.append({'detector': detector, 'risk_domain': domain, 'scenario_type': scenario,
                     'positive_players': len(players), 'detected_players': detected,
                     'false_negatives': len(players) - detected,
                     'recall_at_k': detected / max(len(players), 1)})
    return rows


def _channel_domain_recall(scores, labels, transactions, detector, domain, budget):
    column = f'{domain}_score'
    reviewed = set(scores.sort_values([column, 'player_id'], ascending=[False, True]).head(budget).player_id.astype(str))
    positive = _domain_positive_players(labels, domain)
    observed = transactions[transactions.observed.fillna(False)]
    rows = []
    for channel, group in observed.groupby('channel', sort=True):
        players = set(group.player_id.astype(str))
        channel_positive = positive & players
        detected = channel_positive & reviewed
        rows.append({'detector': detector, 'risk_domain': domain, 'channel': channel,
                     'positive_players': len(channel_positive), 'detected_players': len(detected),
                     'false_negatives': len(channel_positive - reviewed),
                     'recall_at_k': len(detected) / max(len(channel_positive), 1)})
    return rows


def fairness_metrics(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    profiles: pd.DataFrame,
    domain: str,
    group_field: str,
    budget: int,
    small_sample: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit outcomes by protected group without using groups in scoring."""
    if group_field not in {'age_band', 'region'}:
        raise ValueError('Fairness audit supports age_band or region only')
    column = f'{domain}_score'
    ranked = scores.sort_values([column, 'player_id'], ascending=[False, True]).head(budget)
    flagged = set(ranked.player_id.astype(str))
    positive = _domain_positive_players(labels, domain)
    frame = profiles[['player_id', group_field]].drop_duplicates('player_id').copy()
    frame['player_id'] = frame.player_id.astype(str)
    frame['flagged'] = frame.player_id.isin(flagged)
    frame['positive'] = frame.player_id.isin(positive)
    frame['tp'] = frame.flagged & frame.positive
    frame['fp'] = frame.flagged & ~frame.positive
    frame['fn'] = ~frame.flagged & frame.positive
    rows = []
    for group, values in frame.groupby(group_field, dropna=False, sort=True):
        positives = int(values.positive.sum())
        negatives = len(values) - positives
        tp, fp, fn = int(values.tp.sum()), int(values.fp.sum()), int(values.fn.sum())
        rows.append({group_field: group, 'risk_domain': domain, 'players': len(values),
                     'positive_cases': positives, 'flag_rate': values.flagged.mean(),
                     'false_positive_rate': fp / max(negatives, 1),
                     'false_negative_rate': fn / max(positives, 1),
                     'recall': tp / max(positives, 1),
                     'small_sample_warning': len(values) < small_sample or positives < 5})
    groups = pd.DataFrame(rows)
    disparities = []
    for metric in ('flag_rate', 'false_positive_rate', 'false_negative_rate', 'recall'):
        values = groups[metric].dropna()
        positive_values = values[values.gt(0)]
        ratio = values.max() / positive_values.min() if len(positive_values) else np.nan
        disparities.append({'group_field': group_field, 'risk_domain': domain,
                            'metric': metric, 'max_to_min_nonzero_ratio': ratio,
                            'small_sample_warning': bool(groups.small_sample_warning.any())})
    return groups, pd.DataFrame(disparities)


def run_validation_pipeline(
    n_players: int = 300,
    target_events: int | None = None,
    development_seed: int = 142,
    evaluation_seed: int = 1042,
    alert_budget: int = 50,
    threshold: float = 60.0,
) -> ValidationResult:
    """Run reproducible development/held-out assurance with a fixed evaluation population."""
    if development_seed == evaluation_seed:
        raise ValueError('Development and evaluation seeds must differ')
    target_events = target_events or n_players * 20
    # The development population proves separation and is retained in metadata;
    # no held-out outcome is used to alter thresholds or detector configuration.
    development = generate(n_players=n_players, target_events=target_events, seed=development_seed)
    held_out = generate(n_players=n_players, target_events=target_events, seed=evaluation_seed)
    development_scores = risk_detector_scores(development.profiles, development.transactions)
    score_tables = risk_detector_scores(held_out.profiles, held_out.transactions)

    overall, scenarios, channels = [], [], []
    for detector in DETECTOR_ORDER:
        scores = score_tables[detector]
        for domain in RISK_DOMAINS:
            overall.append({'detector': detector, 'risk_domain': domain,
                            **classification_metrics(scores, held_out.scenario_labels, domain,
                                                     alert_budget, threshold)})
            scenarios.extend(_scenario_domain_recall(scores, held_out.scenario_labels,
                                                      detector, domain, alert_budget))
            channels.extend(_channel_domain_recall(scores, held_out.scenario_labels,
                                                    held_out.transactions, detector, domain,
                                                    alert_budget))

    # Evaluate observation loss and identity uncertainty on the same held-out
    # players, labels and review capacity.
    from red_team import visibility_condition
    sensitivity = []
    for condition in ('complete_cross_channel', 'partial_cross_channel',
                      'single_channel', 'uncertain_identity_linkage'):
        changed = visibility_condition(held_out.transactions, condition, seed=evaluation_seed)
        changed_scores = risk_detector_scores(held_out.profiles, changed)[DETECTOR_ORDER[-1]]
        for domain in RISK_DOMAINS:
            metrics = classification_metrics(changed_scores, held_out.scenario_labels, domain,
                                             alert_budget, threshold)
            sensitivity.append({'cohort': condition, 'risk_domain': domain,
                                'players': metrics['population'],
                                'alert_volume': metrics['alert_volume'],
                                'false_positives': metrics['top_k_false_positives'],
                                'false_negatives': metrics['top_k_false_negatives'],
                                'recall_at_k': metrics['recall_at_k']})

    final_scores = score_tables[DETECTOR_ORDER[-1]]
    max_time = pd.to_datetime(held_out.transactions.timestamp).max()
    tenure = (max_time - pd.to_datetime(held_out.profiles.account_open_date)).dt.days
    cohorts = {
        'cold_start_under_90_days': set(held_out.profiles.loc[tenure.lt(90), 'player_id'].astype(str)),
        'benign_high_spending_controls': set(held_out.scenario_labels.loc[
            held_out.scenario_labels.scenario_type.eq('legitimate_high_spender')
            & ~held_out.scenario_labels.ground_truth_label.astype(bool), 'player_id'].astype(str)),
    }
    home = held_out.profiles.set_index('player_id').region
    observed = held_out.transactions[held_out.transactions.observed.fillna(False)].copy()
    changed_geo = observed.region.notna() & observed.player_id.map(home).notna() & observed.region.ne(observed.player_id.map(home))
    risky_geo = set(held_out.scenario_labels.loc[
        held_out.scenario_labels.scenario_type.eq('unusual_geographic_activity')
        & held_out.scenario_labels.ground_truth_label.astype(bool), 'player_id'].astype(str))
    cohorts['observed_geographic_change_without_injected_geo_risk'] = set(observed.loc[changed_geo, 'player_id'].astype(str)) - risky_geo
    for cohort, players in cohorts.items():
        for domain in RISK_DOMAINS:
            column = f'{domain}_score'
            flagged = set(final_scores.sort_values([column, 'player_id'], ascending=[False, True]).head(alert_budget).player_id.astype(str))
            positive = _domain_positive_players(held_out.scenario_labels, domain)
            sensitivity.append({'cohort': cohort, 'risk_domain': domain, 'players': len(players),
                                'alert_volume': len(players & flagged),
                                'false_positives': len((players & flagged) - positive),
                                'false_negatives': len((players & positive) - flagged),
                                'recall_at_k': len(players & positive & flagged) / max(len(players & positive), 1)
                                if players & positive else np.nan})

    age_tables, region_tables, disparity_tables = [], [], []
    for domain in RISK_DOMAINS:
        age, age_disparity = fairness_metrics(final_scores, held_out.scenario_labels,
                                              held_out.profiles, domain, 'age_band', alert_budget)
        region, region_disparity = fairness_metrics(final_scores, held_out.scenario_labels,
                                                    held_out.profiles, domain, 'region', alert_budget)
        age_tables.append(age); region_tables.append(region)
        disparity_tables.extend([age_disparity, region_disparity])

    return ValidationResult(
        overall_metrics=pd.DataFrame(overall), scenario_recall=pd.DataFrame(scenarios),
        channel_recall=pd.DataFrame(channels), sensitivity=pd.DataFrame(sensitivity),
        fairness_by_age=pd.concat(age_tables, ignore_index=True),
        fairness_by_region=pd.concat(region_tables, ignore_index=True),
        disparity_summary=pd.concat(disparity_tables, ignore_index=True),
        metadata={'development_seed': development_seed, 'evaluation_seed': evaluation_seed,
                  'players_per_split': n_players, 'events_per_split': target_events,
                  'alert_budget': alert_budget, 'fixed_threshold': threshold,
                  'development_score_tables': len(development_scores),
                  'fairness_note': 'Equal flag rates do not necessarily imply fair outcomes.',
                  'data_note': 'All people, activity, labels, and results are synthetic.'},
    )


def evaluation_figures(result: ValidationResult):
    """Return Plotly figures backed only by computed validation tables."""
    import plotly.express as px
    overall = result.overall_metrics
    recall = px.bar(overall, x='detector', y='recall_at_k', color='risk_domain', barmode='group',
                    title='Held-out Recall@K by Detector and Risk Domain')
    errors = px.bar(overall, x='detector', y=['top_k_false_positives', 'top_k_false_negatives'],
                    facet_row='risk_domain', barmode='group', title='Held-out Error Counts at Fixed Review Budget')
    fairness = px.bar(result.fairness_by_age, x='age_band', y='recall', color='risk_domain',
                      barmode='group', title='Recall by Age Band (Evaluation Only)')
    return {'recall_at_k': recall, 'error_counts': errors, 'fairness': fairness}


def _main():
    import argparse

    parser = argparse.ArgumentParser(description='Run synthetic held-out Sentinel X validation')
    parser.add_argument('--players', type=int, default=300)
    parser.add_argument('--events', type=int, default=6_000)
    parser.add_argument('--development-seed', type=int, default=142)
    parser.add_argument('--evaluation-seed', type=int, default=1042)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=60.0)
    args = parser.parse_args()
    result = run_validation_pipeline(
        n_players=args.players, target_events=args.events,
        development_seed=args.development_seed, evaluation_seed=args.evaluation_seed,
        alert_budget=args.budget, threshold=args.threshold,
    )
    columns = [
        'detector', 'risk_domain', 'precision', 'recall', 'f1',
        'precision_at_k', 'recall_at_k', 'top_k_false_positives',
        'top_k_false_negatives', 'alert_volume', 'review_workload',
    ]
    print('Synthetic held-out evaluation (not an OLG performance claim)')
    print(result.metadata)
    print(result.overall_metrics[columns].to_string(index=False))


if __name__ == '__main__':
    _main()
