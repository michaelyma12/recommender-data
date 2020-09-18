#!/usr/bin/env bash
echo "Updating apt-get ..."
apt-get update

echo "Setting up python environment ..."
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-latest-hadoop2.jar -P /usr/lib/spark/jars/
gsutil cp gs://recommender-amazon-1/environment.yml .
/opt/conda/miniconda3/bin/conda env create -f environment.yml

echo "Setting up credentials ..."
cd /
mkdir -p recommender-data/models/candidate_generation
gsutil cp -r gs://recommender-amazon-1/.gcp recommender-data/
gsutil cp -r gs://recommender-amazon-1/pipeline recommender-data/

echo "Activating conda environment as default ..."
echo "export PYTHONPATH=/recommender-data:${PYTHONPATH}" | tee -a /etc/profile.d/effective-python.sh ~/.bashrc
echo "export PYSPARK_PYTHON=/opt/conda/miniconda3/envs/recommender-data/bin/python3.7" >> /etc/profile.d/effective-python.sh
