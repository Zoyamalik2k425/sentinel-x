"""Defensive adversarial evaluation on fictional Sentinel X observations.

The transformations below are deliberately abstract stressors for synthetic
data. They are not operational instructions and must not be interpreted as
guidance for avoiding financial-crime or player-protection controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_gen import OperationalDataset, generate
from signals import GROUND_TRUTH_COLUMNS
from validate import (
    detector_scores,
    evaluate_ranked_scores,
    recall_by_coverage,
    recall_by_scenario,
)


BEHAVIOR_LEVELS = {
    0: 'Obvious unusual behavior',
    1: 'Variable synthetic observations',
    2: 'Abstract multi-day and multi-channel dispersion',
    3: 'Greater resemblance to synthetic historical patterns',
}
VISIBILITY_CONDITIONS = (
    'single_channel',
    'complete_cross_channel',
    'partial_cross_channel',
    'uncertain_identity_linkage',
)
MONEY_COLUMNS = ('deposit', 'wager', 'net_loss')


@dataclass(frozen=True)
class ArenaResult:
    behavioral_metrics: pd.DataFrame
    visibility_metrics: pd.DataFrame
    scenario_recall: pd.DataFrame
    coverage_recall: pd.DataFrame
    metadata: dict[str, int | str]


def _reject_label_leakage(transactions: pd.DataFrame):
    leaked = GROUND_TRUTH_COLUMNS.intersection(transactions.columns)
    if leaked:
        raise ValueError(
            f'Ground-truth columns cannot enter adversarial detector inputs: {sorted(leaked)}'
        )


def _active_synthetic_mask(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    positives = labels[labels.ground_truth_label.astype(bool)].copy()
    starts = pd.to_datetime(positives.set_index('player_id').start_time)
    player_start = transactions.player_id.map(starts)
    return player_start.notna() & pd.to_datetime(transactions.timestamp).ge(player_start)


def behavioral_adaptation(
    transactions: pd.DataFrame,
    scenario_labels: pd.DataFrame,
    level: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply one predefined abstract stress level to fictional positive cases."""
    if level not in BEHAVIOR_LEVELS:
        raise ValueError('level must be one of 0, 1, 2, or 3')
    _reject_label_leakage(transactions)
    output = transactions.copy()
    if level == 0:
        return output
    rng = np.random.default_rng(seed + 101 * level)
    active = _active_synthetic_mask(output, scenario_labels)
    indices = output.index[active]
    original_start = pd.to_datetime(output.timestamp).min()
    original_end = pd.to_datetime(output.timestamp).max()

    # Level 1 changes only the presentation variability of already-injected
    # fictional observations. It creates no new scenario knowledge.
    factors = np.clip(rng.normal(1.0, .18, len(indices)), .65, 1.35)
    for column in MONEY_COLUMNS:
        values = pd.to_numeric(output.loc[indices, column], errors='coerce')
        output.loc[indices, column] = (values * factors).round(2)
    day_offsets = rng.integers(-2, 3, len(indices))
    output.loc[indices, 'timestamp'] = (
        pd.to_datetime(output.loc[indices, 'timestamp'])
        + pd.to_timedelta(day_offsets, unit='D')
    ).clip(original_start, original_end)

    if level >= 2:
        # Defensive abstraction: spread synthetic observations over calendar
        # positions and channel labels without modelling a real tactic.
        extra_offsets = rng.integers(-4, 5, len(indices))
        output.loc[indices, 'timestamp'] = (
            pd.to_datetime(output.loc[indices, 'timestamp'])
            + pd.to_timedelta(extra_offsets, unit='D')
        ).clip(original_start, original_end)
        channels = np.array(['online_casino', 'sportsbook', 'land_casino'])
        output.loc[indices, 'channel'] = rng.choice(channels, size=len(indices), replace=True)

    if level >= 3:
        starts = pd.to_datetime(
            scenario_labels.set_index('player_id').start_time
        )
        player_start = output.player_id.map(starts)
        historical = output[player_start.isna() | pd.to_datetime(output.timestamp).lt(player_start)]
        for column in MONEY_COLUMNS:
            positive_history = historical[pd.to_numeric(historical[column], errors='coerce').gt(0)]
            medians = positive_history.groupby('player_id')[column].median()
            fallback = pd.to_numeric(positive_history[column], errors='coerce').median()
            fallback = 0.0 if pd.isna(fallback) else float(fallback)
            current = pd.to_numeric(output.loc[indices, column], errors='coerce')
            baseline = output.loc[indices, 'player_id'].map(medians).fillna(fallback)
            positive = current.gt(0)
            blended = current.copy()
            blended.loc[positive] = .35 * current.loc[positive] + .65 * baseline.loc[positive]
            output.loc[indices, column] = blended.round(2)

    output['deposit'] = pd.to_numeric(output.deposit, errors='coerce').clip(lower=0)
    output['wager'] = pd.to_numeric(output.wager, errors='coerce').clip(lower=0)
    output['net_loss'] = pd.to_numeric(output.net_loss, errors='coerce').clip(lower=0)
    output['net_loss'] = np.minimum(output.net_loss, output.wager)
    output['week'] = (
        (pd.to_datetime(output.timestamp) - original_start).dt.days // 7
    ).astype(int)
    return output.sort_values(['timestamp', 'tx_id']).reset_index(drop=True)


def visibility_condition(
    transactions: pd.DataFrame,
    condition: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply one channel/identity availability condition to the whole population."""
    if condition not in VISIBILITY_CONDITIONS:
        raise ValueError(f'Unknown visibility condition: {condition}')
    _reject_label_leakage(transactions)
    output = transactions.copy()
    if condition == 'complete_cross_channel':
        return output
    if condition == 'single_channel':
        output.loc[~output.channel.eq('online_casino'), 'observed'] = False
        return output
    rng = np.random.default_rng(seed + 809)
    if condition == 'partial_cross_channel':
        candidates = output.index[~output.channel.eq('online_casino')]
        hidden = candidates[rng.random(len(candidates)) < .45]
        output.loc[hidden, 'observed'] = False
        return output

    candidates = output.index[output.channel.isin(['land_casino', 'sportsbook'])]
    uncertain = candidates[rng.random(len(candidates)) < .30]
    output.loc[uncertain, 'player_id'] = 'UNLINKED-' + output.loc[uncertain, 'tx_id'].astype(str)
    return output


def stress_test(tx, ground_truth, level):
    """Backward-compatible missing-visibility stress path used by Phase 9."""
    if level not in range(4):
        raise ValueError('level must be one of 0, 1, 2, or 3')
    output = tx.copy()
    if level == 0:
        return output
    risky = set(ground_truth.loc[ground_truth.ground_truth_label, 'player_id'])
    mask = output.player_id.isin(risky) & (output.week >= output.week.max() - 4)
    if level >= 1:
        output.loc[mask & output.channel.eq('land_casino'), 'observed'] = False
    if level >= 2:
        output.loc[mask & output.channel.eq('sportsbook'), 'observed'] = False
    if level >= 3:
        ids = output.index[mask & output.channel.eq('online_casino')]
        output.loc[ids[::2], 'observed'] = False
    return output


def _condition_evaluation(
    dataset: OperationalDataset,
    transactions: pd.DataFrame,
    budget: int,
    split: str,
    experiment: str,
    condition: str,
):
    metrics_rows, scenario_rows, coverage_rows = [], [], []
    for detector, ranking in detector_scores(dataset.profiles, transactions):
        metrics = evaluate_ranked_scores(ranking, dataset.scenario_labels, budget)
        metrics_rows.append({
            'dataset_split': split,
            'experiment': experiment,
            'condition': condition,
            'detector': detector,
            **metrics,
        })
        scenario = recall_by_scenario(ranking, dataset.scenario_labels, budget)
        scenario.insert(0, 'detector', detector)
        scenario.insert(0, 'condition', condition)
        scenario.insert(0, 'experiment', experiment)
        scenario.insert(0, 'dataset_split', split)
        scenario_rows.append(scenario)
        coverage = recall_by_coverage(
            ranking, dataset.scenario_labels, transactions, budget
        )
        coverage.insert(0, 'detector', detector)
        coverage.insert(0, 'condition', condition)
        coverage.insert(0, 'experiment', experiment)
        coverage.insert(0, 'dataset_split', split)
        coverage_rows.append(coverage)
    return metrics_rows, scenario_rows, coverage_rows


def run_adversarial_arena(
    n_players: int = 300,
    target_events: int | None = None,
    development_seed: int = 142,
    evaluation_seed: int = 1042,
    alert_budget: int = 50,
) -> ArenaResult:
    """Run both experiments on separate development and held-out populations."""
    if development_seed == evaluation_seed:
        raise ValueError('Development and evaluation seeds must be different')
    if alert_budget < 1:
        raise ValueError('alert_budget must be positive')
    target_events = target_events or n_players * 20
    datasets = {
        'development': generate(
            n_players=n_players, seed=development_seed, target_events=target_events
        ),
        'held_out_evaluation': generate(
            n_players=n_players, seed=evaluation_seed, target_events=target_events
        ),
    }
    behavior_metrics, visibility_metrics = [], []
    scenario_tables, coverage_tables = [], []
    for split, dataset in datasets.items():
        for level, label in BEHAVIOR_LEVELS.items():
            transformed = behavioral_adaptation(
                dataset.transactions, dataset.scenario_labels, level,
                seed=(development_seed if split == 'development' else evaluation_seed),
            )
            metrics, scenarios, coverage = _condition_evaluation(
                dataset, transformed, alert_budget, split,
                'behavioral_adaptation', f'level_{level}',
            )
            for row in metrics:
                row['complexity_level'] = level
                row['condition_label'] = label
            behavior_metrics.extend(metrics)
            scenario_tables.extend(scenarios)
            coverage_tables.extend(coverage)

        for condition in VISIBILITY_CONDITIONS:
            transformed = visibility_condition(
                dataset.transactions, condition,
                seed=(development_seed if split == 'development' else evaluation_seed),
            )
            metrics, scenarios, coverage = _condition_evaluation(
                dataset, transformed, alert_budget, split,
                'cross_channel_visibility', condition,
            )
            visibility_metrics.extend(metrics)
            scenario_tables.extend(scenarios)
            coverage_tables.extend(coverage)

    return ArenaResult(
        behavioral_metrics=pd.DataFrame(behavior_metrics),
        visibility_metrics=pd.DataFrame(visibility_metrics),
        scenario_recall=pd.concat(scenario_tables, ignore_index=True),
        coverage_recall=pd.concat(coverage_tables, ignore_index=True),
        metadata={
            'development_seed': development_seed,
            'evaluation_seed': evaluation_seed,
            'players_per_split': n_players,
            'events_per_split': target_events,
            'alert_budget': alert_budget,
            'purpose': 'defensive synthetic evaluation only',
        },
    )
