import pandas as pd
import pytest

from data_gen import generate
from signals import GROUND_TRUTH_COLUMNS
from validate import (
    DETECTOR_ORDER,
    RISK_DOMAINS,
    classification_metrics,
    evaluation_figures,
    risk_detector_scores,
    run_validation_pipeline,
)


def test_detector_scores_are_separate_from_labels_and_demographics():
    dataset = generate(n_players=60, target_events=1_200, seed=51)
    tables = risk_detector_scores(dataset.profiles, dataset.transactions)

    assert tuple(tables) == DETECTOR_ORDER
    for table in tables.values():
        assert set(table) == {'player_id', 'aml_fraud_score', 'safer_gambling_score'}
        assert GROUND_TRUTH_COLUMNS.isdisjoint(table.columns)
        assert table.aml_fraud_score.between(0, 100).all()
        assert table.safer_gambling_score.between(0, 100).all()


def test_domains_are_evaluated_independently():
    scores = pd.DataFrame({
        'player_id': ['A', 'B', 'C'],
        'aml_fraud_score': [90, 10, 20],
        'safer_gambling_score': [10, 90, 20],
    })
    labels = pd.DataFrame({
        'player_id': ['A', 'B'], 'scenario_type': ['aml_case', 'sg_case'],
        'start_time': pd.to_datetime(['2026-01-01', '2026-01-01']),
        'end_time': pd.to_datetime(['2026-02-01', '2026-02-01']),
        'target_risk_category': ['aml_fraud', 'safer_gambling'],
        'ground_truth_label': [True, True],
    })
    aml = classification_metrics(scores, labels, 'aml_fraud', budget=1, threshold=60)
    sg = classification_metrics(scores, labels, 'safer_gambling', budget=1, threshold=60)
    assert aml['true_positives'] == sg['true_positives'] == 1
    assert aml['false_negatives'] == sg['false_negatives'] == 0


@pytest.fixture(scope='module')
def report():
    return run_validation_pipeline(n_players=80, target_events=1_600,
                                   development_seed=61, evaluation_seed=71,
                                   alert_budget=15)


def test_pipeline_has_fixed_population_budget_sensitivity_and_fairness(report):
    assert set(report.overall_metrics.detector) == set(DETECTOR_ORDER)
    assert set(report.overall_metrics.risk_domain) == set(RISK_DOMAINS)
    assert report.overall_metrics.population.nunique() == 1
    assert report.overall_metrics.alert_budget.eq(15).all()
    assert report.overall_metrics.recall_at_k.between(0, 1).all()
    assert {'partial_cross_channel', 'uncertain_identity_linkage',
            'cold_start_under_90_days', 'benign_high_spending_controls'}.issubset(
                set(report.sensitivity.cohort)
            )
    assert {'flag_rate', 'false_positive_rate', 'false_negative_rate',
            'recall', 'small_sample_warning'}.issubset(report.fairness_by_age.columns)
    assert report.metadata['development_seed'] != report.metadata['evaluation_seed']
    assert 'Equal flag rates' in report.metadata['fairness_note']


def test_evaluation_plots_are_built_from_computed_tables(report):
    figures = evaluation_figures(report)
    assert set(figures) == {'recall_at_k', 'error_counts', 'fairness'}
    assert all(figure.data for figure in figures.values())
