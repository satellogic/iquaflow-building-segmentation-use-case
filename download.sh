#!/bin/bash

TO_PATH=./dataset
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-building-segmentation-use-case/building_footprint_dataset.zip -O $TO_PATH/building_footprint_dataset.zip
unzip $TO_PATH/building_footprint_dataset.zip -d $TO_PATH