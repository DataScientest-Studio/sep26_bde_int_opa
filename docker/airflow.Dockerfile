# Airflow image for the OPA batch pipeline: official image + the libraries our scripts import.
# Pinning apache-airflow in the same pip call keeps pip from up/downgrading Airflow itself.
ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.10
ARG AIRFLOW_VERSION

# scikit-learn and joblib: the predict and train steps load and save the models. The versions must match what
# a normal PC installs from requirements.txt, because models trained in the notebooks are loaded here:
#   - scikit-learn pinned to the same version as requirements.txt
#   - numpy 2: a model saved with numpy 2 does not load under the image's default numpy 1.26
#     (tested: the gradient-boosting models fail), and pandas >= 2.2.2 is the first that works with numpy 2
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    "pymongo>=4,<5" "python-dotenv>=1,<2" requests psycopg2-binary \
    "numpy>=2,<3" "pandas>=2.2.2,<3" "scikit-learn==1.7.2" joblib
