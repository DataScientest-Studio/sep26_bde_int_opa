"""OPA batch pipeline (Stage 3): Binance -> bronze -> silver -> gold, every 15 minutes.

Owner: Talgat

Each task runs one of the existing project scripts unchanged (BashOperator), so the
pipeline in Airflow is exactly the pipeline you can run by hand from the README.
A script that fails exits with a non-zero code, which marks the task as failed;
the downstream tasks then do not run and Airflow retries it.

    extract_rest_bronze   Binance REST  -> Mongo raw_klines            (src/data/extract_rest.py)
    transform_silver      Mongo         -> Postgres fact_kline          (src/data/transform.py)
    build_features_gold   fact_kline    -> gold_features + outcomes     (src/features/build_features.py)
    check_quality         freshness + gold coverage, fails loudly       (src/pipeline/check_quality.py)
    predict               active models -> Postgres predictions          (src/models/predict_model.py --all)

Schedule: minute 1, 16, 31, 46 - one minute after each 15m Binance candle closes.
predict runs only after the quality gate passed, so the API never serves a prediction made on
stale or incomplete data. Before the first training there is no active model: predict then
writes nothing and still succeeds. The models themselves are retrained by opa_retrain_weekly.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

OPA_HOME = "/opt/airflow/opa"   # src/ and sql/ are mounted here (docker-compose.yml)

default_args = {
    "owner": "opa",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}

with DAG(
    dag_id="opa_pipeline_15m",
    description="Binance -> Mongo (bronze) -> Postgres (silver, gold) every 15 minutes",
    schedule="1,16,31,46 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,          # never backfill old runs; extract_rest.py fills gaps by itself
    max_active_runs=1,      # runs never overlap
    default_args=default_args,
    tags=["opa", "batch"],
    doc_md=__doc__,
) as dag:

    def script(task_id, path):
        return BashOperator(task_id=task_id, bash_command=f"cd {OPA_HOME} && python {path}")

    extract = script("extract_rest_bronze", "src/data/extract_rest.py")
    transform = script("transform_silver", "src/data/transform.py")
    features = script("build_features_gold", "src/features/build_features.py")
    quality = script("check_quality", "src/pipeline/check_quality.py")
    predict = script("predict", "src/models/predict_model.py --all")

    extract >> transform >> features >> quality >> predict
