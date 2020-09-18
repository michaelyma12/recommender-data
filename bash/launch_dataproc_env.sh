#!/usr/bin/env bash
python generate_custom_image.py \
    --image-name recommender-data-1 \
    --dataproc-version "1.5.10-debian10" \
    --customization-script startup_script/recommender_data_conda.sh \
    --zone europe-west3-a \
    --gcs-bucket gs://recommender-amazon-1