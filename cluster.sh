#!/bin/bash

INPUT=$1

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE}")")"

cd ${SCRIPT_DIR}

source .venv/bin/activate

if [ ! -n "$INPUT" ]; then
    echo "Specify input file e.g. example-input-queries.csv"
    exit 1
fi
python cluster.py \
  --input "$INPUT" \
  --output-summary output/summary.csv \
  --output-samples output/samples.csv \
  --use-umap-before-clustering \
  --umap-cluster-n-neighbors 3 \
  --umap-cluster-min-dist 0.0 \
  --umap-cluster-metric cosine \
  --min-cluster-size 2 \
  --min-samples 3 \
  --cluster-selection-epsilon 0.2 \
  --cluster-selection-method eom \
  --alpha 1.0 \
  --num-samples-to-show 10 \
  --visualize-umap output/umap_viz.png 
