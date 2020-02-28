#!/bin/bash

# current script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT="$DIR/../../../.."

cp $ROOT/README.md .
cp $ROOT/LICENSE .
cp $ROOT/tensorflow_model_analysis/static/vulcanized_tfma.js dist/
