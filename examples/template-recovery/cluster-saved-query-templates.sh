#!/bin/bash

# Example based on lustre/docs/TEMPLATE-RECOVERY.md

INPUT=$1

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE}")")"

cd ${SCRIPT_DIR}

source .venv/bin/activate

if [ ! -n "$INPUT" ]; then
    echo "Specify input file e.g. example/example-input-queries.csv"
    exit 1
fi
# --min-samples 35
# --min-cluster-size 5–20
# --cluster-selection-epsilon 0.1–0.3
python cluster.py \
  --input "$INPUT" \
  --output-summary output/summary.csv \
  --output-samples output/samples.csv \
  --num-samples-to-show 10 \
  --visualize-umap output/umap_viz.png \
  --visualize-tsne output/tsne_viz.png \
  --use-umap-before-clustering \
  --min-cluster-size 5 \
  --min-samples 3 \
  --cluster-selection-epsilon 0.1
