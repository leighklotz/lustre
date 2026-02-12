#!/bin/bash

python cluster.py --input data/ALL_SEARCH.tsv --output-summary output/summary.csv --output-samples output/cluster-samples.csv --min-samples 5   --visualize-tsne output/tsne_plot.png   --visualize-umap output/umap_plot.png --min-cluster-size 10 --use-umap-before-clustering  --cluster-selection-method leaf
