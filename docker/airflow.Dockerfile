# Airflow image for the OPA batch pipeline: official image + the libraries our scripts import.
# Pinning apache-airflow in the same pip call keeps pip from up/downgrading Airflow itself.
ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.10
ARG AIRFLOW_VERSION

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    "pymongo>=4,<5" "python-dotenv>=1,<2" pandas requests psycopg2-binary
