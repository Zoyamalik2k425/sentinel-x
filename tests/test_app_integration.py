from pathlib import Path

from streamlit.testing.v1 import AppTest

from data_gen import OperationalDataset, generate
from signals import GROUND_TRUTH_COLUMNS


def test_operational_generator_has_named_three_table_contract():
    result = generate(n_players=40, weeks=8, target_events=800, seed=91)

    assert isinstance(result, OperationalDataset)
    assert len(result.profiles) == 40
    assert len(result.transactions) > 0
    assert {'scenario_type', 'ground_truth_label'}.issubset(result.scenario_labels.columns)
    assert GROUND_TRUTH_COLUMNS.isdisjoint(result.profiles.columns)
    assert GROUND_TRUTH_COLUMNS.isdisjoint(result.transactions.columns)


def test_streamlit_app_loads_without_data_contract_exception():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    assert len(app.tabs) == 4
