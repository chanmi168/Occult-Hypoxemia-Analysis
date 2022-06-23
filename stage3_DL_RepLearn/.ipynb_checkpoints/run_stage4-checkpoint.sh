#!/bin/bash

set -e
set -u

inputdir="../../data/stage3_DL_RepLearn/"
outputdir="../../data/stage3_DL_RepLearn/"
training_params_file="training_params_dummy.json"

echo "===================================== running stage 4 [regression]  ====================================="
mkdir -p $outputdir
python DL_RepLearn.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
