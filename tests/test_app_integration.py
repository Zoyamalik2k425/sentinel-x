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


def test_dashboard_factory_returns_complete_scored_payload(monkeypatch):
    # Importing the app also exercises its default pipeline in Streamlit's
    # bare test context; the uncached builder then verifies the exact contract
    # that previously returned None.
    monkeypatch.setenv('SENTINEL_X_DEFAULT_PLAYERS', '100')
    import app as dashboard_app

    result = dashboard_app._build_dashboard_data(40, 93, True)

    assert isinstance(result, dashboard_app.DashboardData)
    assert len(result.synthetic.players) == 40
    assert not result.events.empty
    assert not result.scored.empty
    assert {
        'player_id', 'aml_fraud_score', 'safer_gambling_score',
        'signal_contributions', 'supporting_transaction_ids',
    }.issubset(result.scored.columns)
    assert GROUND_TRUTH_COLUMNS.isdisjoint(result.scored.columns)


def test_streamlit_app_loads_without_data_contract_exception(monkeypatch):
    monkeypatch.setenv('SENTINEL_X_DEFAULT_PLAYERS', '100')
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    navigation = next(radio for radio in app.radio if radio.label == 'Workspace')
    assert navigation.options == [
        'Overview', 'Investigation Queue', 'Player 360',
        'Network Intelligence', 'Validation & Fairness', 'Red vs Blue Arena',
    ]
    assert any('Risk distribution' in subheader.value for subheader in app.subheader)

    navigation.set_value('Player 360').run(timeout=30)
    assert not app.exception
    button_labels = {button.label for button in app.button}
    assert {'Confirm for Escalation', 'Dismiss Review', 'Save Note'}.issubset(button_labels)
    markdown = {element.value for element in app.markdown}
    assert '#### EVIDENCE FOR CONCERN' in markdown
    assert '#### EVIDENCE AGAINST OR ALTERNATIVE EXPLANATIONS' in markdown
    assert '#### Recommended human-review actions' in markdown

    for page in [
        'Investigation Queue', 'Network Intelligence', 'Validation & Fairness',
        'Red vs Blue Arena',
    ]:
        navigation = next(radio for radio in app.radio if radio.label == 'Workspace')
        navigation.set_value(page).run(timeout=30)
        assert not app.exception
        if page == 'Investigation Queue':
            assert app.dataframe
            queue = app.dataframe[0].value
            assert len(queue) > 0
            assert {
                'priority_rank', 'aml_fraud_score', 'safer_gambling_score',
                'primary_alert_reason', 'investigation_status',
            }.issubset(queue.columns)
