import pandas as pd
import pytest

from data_gen import generate
from models import predict
from red_team import stress_test
from scoring import explain, score
from signals import GROUND_TRUTH_COLUMNS, compute
from validate import evaluate


@pytest.fixture(scope='module')
def dataset():
    return generate(n_players=100, weeks=16, seed=42)


def test_generation_is_reproducible_and_labels_are_separate(dataset):
    profiles, transactions, ground_truth = dataset
    again = generate(n_players=100, weeks=16, seed=42)
    assert len(profiles) == 100
    assert len(transactions) > 100
    assert set(transactions.channel.unique()) == {'online_casino', 'sportsbook', 'land_casino'}
    assert GROUND_TRUTH_COLUMNS.isdisjoint(profiles.columns)
    assert {'scenario_type', 'ground_truth_label'}.issubset(ground_truth.columns)
    for actual, repeated in zip(dataset, again):
        pd.testing.assert_frame_equal(actual, repeated)


def test_operational_pipeline_has_separate_risk_outputs(dataset):
    profiles, transactions, _ = dataset
    features = compute(profiles, transactions)
    outputs = predict(features)
    scored = score(features)
    assert len(scored) == len(profiles)
    assert {'aml_fraud_risk', 'safer_gambling_risk'}.issubset(outputs.columns)
    assert scored.aml_fraud_score.between(0, 100).all()
    assert scored.safer_gambling_score.between(0, 100).all()
    assert scored.priority.between(0, 100).all()
    assert GROUND_TRUTH_COLUMNS.isdisjoint(scored.columns)
    assert len(explain(scored.iloc[0])) == 3


def test_evaluation_and_stress_paths_use_held_labels(dataset):
    profiles, transactions, ground_truth = dataset
    report = evaluate(profiles, transactions, ground_truth, k=20)
    assert len(report) == 3
    assert report.recall_at_k.between(0, 1).all()
    assert len(stress_test(transactions, ground_truth, 3)) == len(transactions)


def test_feature_engineering_rejects_label_leakage(dataset):
    profiles, transactions, ground_truth = dataset
    contaminated = profiles.merge(ground_truth, on='player_id')
    with pytest.raises(ValueError, match='Ground-truth columns'):
        compute(contaminated, transactions)


def test_stress_level_is_validated(dataset):
    _, transactions, ground_truth = dataset
    with pytest.raises(ValueError, match='level'):
        stress_test(transactions, ground_truth, 4)
