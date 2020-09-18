#!/usr/bin/env bash
echo "Setting up google storage api ..."
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-latest-hadoop2.jar -P /usr/lib/spark/jars/
cd /
mkdir -p recommender-data/models/candidate_generation
gsutil cp -r gs://recommender-amazon-1/.gcp recommender-data/
gsutil cp -r gs://recommender-amazon-1/pipeline recommender-data/

echo "Setting up python packages ..."
apt-get -y update
pip install numpy
pip install pandas
pip install google-cloud-storage
