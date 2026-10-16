import pandas as pd
import pytest

from data_gen import generate
from red_team import behavioral_adaptation, run_adversarial_arena, visibility_condition
from signals import GROUND_TRUTH_COLUMNS
from validate import detector_scores, evaluate_ranked_scores


@pytest.fixture(scope='module')
def dataset():
    return generate(n_players=60, target_events=1_200, seed=701)


def test_behavioral_levels_are_reproducible_label_free_and_non_mutating(dataset):
    original = dataset.transactions.copy(deep=True)
    first = behavioral_adaptation(original, dataset.scenario_labels, level=3, seed=81)
    second = behavioral_adaptation(original, dataset.scenario_labels, level=3, seed=81)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(original, dataset.transactions)
    assert GROUND_TRUTH_COLUMNS.isdisjoint(first.columns)
    assert len(first) == len(original)
    assert first.deposit.ge(0).all()
    assert first.net_loss.le(first.wager).all()


def test_visibility_conditions_preserve_population_and_express_uncertainty(dataset):
    single = visibility_condition(dataset.transactions, 'single_channel', seed=9)
    uncertain = visibility_condition(dataset.transactions, 'uncertain_identity_linkage', seed=9)

    assert len(single) == len(dataset.transactions) == len(uncertain)
    assert not single.loc[~single.channel.eq('online_casino'), 'observed'].any()
    assert uncertain.player_id.astype(str).str.startswith('UNLINKED-').any()
    assert GROUND_TRUTH_COLUMNS.isdisjoint(uncertain.columns)


def test_arena_uses_separate_populations_and_fixed_workload():
    result = run_adversarial_arena(
        n_players=50, target_events=1_000,
        development_seed=17, evaluation_seed=29, alert_budget=10,
    )
    assert set(result.behavioral_metrics.dataset_split) == {'development', 'held_out_evaluation'}
    assert set(result.behavioral_metrics.detector) == {
        'Simple Baseline Rules', 'Blue Detector V1', 'Blue Detector V2'
    }
    assert result.behavioral_metrics.alert_budget.eq(10).all()
    assert result.behavioral_metrics.review_workload.eq(10).all()
    assert result.behavioral_metrics.recall_at_k.between(0, 1).all()
    assert set(result.visibility_metrics.condition) == {
        'single_channel', 'complete_cross_channel',
        'partial_cross_channel', 'uncertain_identity_linkage',
    }
    assert not result.scenario_recall.empty
    assert not result.coverage_recall.empty

    held_out = generate(n_players=50, target_events=1_000, seed=29)
    direct = {
        name: evaluate_ranked_scores(ranking, held_out.scenario_labels, 10)
        for name, ranking in detector_scores(held_out.profiles, held_out.transactions)
        if name in {'Simple Baseline Rules', 'Blue Detector V2'}
    }
    level_zero = result.behavioral_metrics[
        result.behavioral_metrics.dataset_split.eq('held_out_evaluation')
        & result.behavioral_metrics.condition.eq('level_0')
    ].set_index('detector')
    for name, metrics in direct.items():
        assert level_zero.loc[name, 'recall_at_k'] == metrics['recall_at_k']
        assert level_zero.loc[name, 'precision_at_k'] == metrics['precision_at_k']


def test_arena_requires_distinct_development_and_evaluation_seeds():
    with pytest.raises(ValueError, match='different'):
        run_adversarial_arena(n_players=20, target_events=400,
                              development_seed=4, evaluation_seed=4, alert_budget=5)
