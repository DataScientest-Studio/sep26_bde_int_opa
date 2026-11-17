"""OPA weekly retraining (Stage 3): retrain both models on the newest data and serve them.

Owner: Dennis

    train      gold_features + outcomes -> new models, registered and activated   (src/models/train_model.py --activate)
    predict    active models -> Postgres predictions, so the API serves the new models at once

train_model.py compares the candidates on the validation part, refits the best one per task and
symbol, saves it under models/<task>/<symbol>/ and switches the active model in model_registry.
The previous model stays in the registry, so switching back is one UPDATE.

Schedule: Sunday 03:30 UTC (a quiet time; any time works). Start it by hand with the
"Trigger DAG" button after the first data load, so the 15-minute pipeline has models to use.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

OPA_HOME = "/opt/airflow/opa"   # src/, sql/ and models/ are mounted here (docker-compose.yml)

default_args = {
    "owner": "opa",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="opa_retrain_weekly",
    description="Retrain, register and activate the direction and volatility models",
    schedule="30 3 * * 0",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["opa", "ml"],
    doc_md=__doc__,
) as dag:

    def script(task_id, command):
        return BashOperator(task_id=task_id, bash_command=f"cd {OPA_HOME} && python {command}")

    train = script("train", "src/models/train_model.py --activate")
    predict = script("predict", "src/models/predict_model.py --all")

    train >> predict
