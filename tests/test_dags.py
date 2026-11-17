"""DAG structure tests. They need Airflow, so they run inside the Airflow container and are skipped elsewhere:

    docker compose exec airflow-scheduler bash -c "cd /opt/airflow && python -m pytest -q /opt/airflow/opa/tests/test_dags.py"
"""
from pathlib import Path

import pytest

# "airflow.models", not "airflow": the repo's own airflow/ folder would otherwise count as the package
pytest.importorskip("airflow.models")
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "airflow" / "dags"


@pytest.fixture(scope="module")
def bag():
    return DagBag(dag_folder=str(DAGS), include_examples=False)


def test_dags_load_without_errors(bag):
    assert bag.import_errors == {}
    assert {"opa_pipeline_15m", "opa_retrain_weekly"} <= set(bag.dag_ids)


def test_pipeline_order_ends_with_predict(bag):
    dag = bag.get_dag("opa_pipeline_15m")
    order = ["extract_rest_bronze", "transform_silver", "build_features_gold", "check_quality", "predict"]
    for upstream, downstream in zip(order, order[1:]):
        assert dag.get_task(downstream).upstream_task_ids == {upstream}


def test_retrain_trains_then_predicts(bag):
    dag = bag.get_dag("opa_retrain_weekly")
    assert dag.get_task("predict").upstream_task_ids == {"train"}
    assert "--activate" in dag.get_task("train").bash_command
