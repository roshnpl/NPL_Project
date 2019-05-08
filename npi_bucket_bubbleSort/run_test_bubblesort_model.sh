#!/bin/sh

THIS_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${THIS_DIR}/data
MODEL_OUTPUT=${1:-${DATA_DIR}/bubblesort.model}
NUM_TEST=${2:-100}

export PYTHONPATH=${THIS_DIR}
cd "$THIS_DIR"

mkdir -p "$DATA_DIR"

echo python3 npi/bubble/test_model.py "$MODEL_OUTPUT" "$NUM_TEST"
python3 npi/bubble/test_model.py "$MODEL_OUTPUT" "$NUM_TEST"
