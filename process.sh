#!/bin/bash

# install python dependencies
sudo apt update && sudo apt install -y libsndfile1
pip install librosa pretty_midi

# run processing script
python process_data.py

# gsutil cp /tmp/*.tfrecords gs://drums-bucket/
