# Query Cluster

This project clusters sanitized queries into similarity groups and gives aggregate statistics and one or more sample queries.

# Installation and Quick Start
```bash
$ git clone https://github.com/leighklotz/lustre/
$ cd lustre
$ mkdir output
$ python --version
Python 3.12.3
$ python3 -m venv .venv
$ . .venv/bin/activate
(.venv) $ pip install -r requirements.txt 
(.venv) $ python cluster.py --input example-input-queries.csv --output-summary output/summary.csv --output-samples output/cluster-samples.csv
```

## Input `queries.csv` format:
Users is space-separated.

```csv
query,count,runtime,users
"index=web sourcetype=apache_error warn",1434,7,"user1 user3"
"stats count by user",100,5,"user2 user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1 user5"
```

## Summary File Output fields
- `cluster`
- `cluster_size`
- `centroid_query`
- `cluster_total_tunetime`
- `cluster_total_runcount`
- `cluster_all_users`

(`cluster` is -1 for outlier cluster)


## Samples File Output Fields
- `cluster`
- `cluster_size`
- `sample_type`
- `distance_from_centroid`
- `query`

(`sample_type` is one of `median` | `edge` | `centroid` | `diverse`)

## Cluster Summary Output
These are the clusters. There is one sample query per cluster.

```csv
cluster,cluster_size,centroid_query,cluster_total_runtime,cluster_total_runcount,cluster_all_users
-1,2,"stats sum(bytes) as b by host,user",670,8,user13 user21 user3 user4
0,4,index=web sourcetype=apache_error error,2134,15,user1 user20 user25 user3 user6
1,4,index=main sourcetype=syslog error,730,9,user11 user16 user19 user2 user22 user3 user5
2,4,search index=main sourcetype=access_combined status=400,1621,45,AAA BBB user1 user17 user2 user4 user5 user8
3,2,search index=web sourcetype=apache_access status=403,1270,11,user1 user15 user24 user5
4,3,search index=web sourcetype=apache_access status=400,1570,14,user12 user14 user2 user23 user4 user5
5,2,"stats count by user, host",400,9,user2 user4 user7
6,2,stats count(user) by host,370,6,user18 user4 user5 user9
```

## Cluster Samples Output
Below are the selected queries from each cluster. There are three sample queries per cluster.

When you specify `--output-samples <filepath>` it outputs some sample
queries from each cluster to a separate file, so you can inspect the
queries extracted to each cluster more closely.

```csv
cluster,cluster_size,sample_type,distance_from_centroid,query
-1,2,centroid,1.4513,"stats sum(bytes) as b by host,user"
-1,2,edge,1.4513,stats sum(bytes) by host
0,4,centroid,0.4100,index=web sourcetype=apache_error error
0,4,edge,0.6737,index=web sourcetype=apache_error warn
0,4,median,0.5722,index=web sourcetype=apache_error debug
1,4,centroid,0.4200,index=main sourcetype=syslog error
1,4,edge,0.6807,index=main sourcetype=syslog warn
1,4,median,0.5992,index=main sourcetype=syslog info
2,4,centroid,0.0947,search index=main sourcetype=access_combined status=400
2,4,edge,0.3763,search index=main sourcetype=access_combined status=5001
2,4,median,0.3629,search index=main sourcetype=access_combined status=404
3,2,centroid,0.1001,search index=web sourcetype=apache_access status=403
3,2,edge,0.1001,search index=web sourcetype=apache_access status=404
4,3,centroid,0.0503,search index=web sourcetype=apache_access status=400
4,3,edge,0.0845,search index=web sourcetype=apache_access status=500
4,3,median,0.0761,search index=web sourcetype=apache_access status=401
5,2,centroid,1.4877,"stats count by user, host"
5,2,edge,1.4877,stats count by user
6,2,centroid,0.3422,stats count(user) by host
6,2,edge,0.3422,stats count(host) by user
```

# Cluster Tuning Parameters

## Controlling Cluster Size and Tightness

The following parameters control how HDBSCAN clusters your queries. Use these to adjust cluster size, tightness, and the number of clusters produced.

### Core Parameters

**`--min-cluster-size` (default: 2)**
- Minimum number of queries required to form a cluster
- **To get MORE clusters**: Decrease this value (e.g., `--min-cluster-size 2`)
- **To get FEWER clusters**: Increase this value (e.g., `--min-cluster-size 10`)
- Queries that don't meet this threshold become outliers (cluster -1)

**`--min-samples` (default: 1)**
- Number of samples in a neighborhood for a point to be considered a core point
- **To get TIGHTER clusters**: Increase this value (e.g., `--min-samples 5` to `--min-samples 10`)
- **To get LOOSER clusters**: Keep at 1 or use 2
- Higher values make clustering more conservative and noise-resistant

**`--cluster-selection-epsilon` (default: 0.0)**
- Distance threshold below which clusters will be merged
- **To get FEWER, LARGER clusters**: Increase this value (e.g., `--cluster-selection-epsilon 0.2` to `--cluster-selection-epsilon 0.5`)
- **To get MORE, SMALLER clusters**: Keep at 0.0
- Useful for consolidating very similar clusters

**`--cluster-selection-method` (default: 'eom')**
- Method for selecting clusters from the condensed tree
- **Options**: `eom` (Excess of Mass) or `leaf`
- **For TIGHTER, MORE UNIFORM clusters**: Use `--cluster-selection-method leaf`
- **For FLEXIBLE, VARYING DENSITY clusters**: Use `--cluster-selection-method eom` (default)

**`--alpha` (default: 1.0)**
- Conservativeness parameter for cluster selection
- **To get TIGHTER, MORE CONSERVATIVE clusters**: Increase (e.g., `--alpha 1.5` to `--alpha 2.0`)
- **To get LOOSER clusters**: Decrease (e.g., `--alpha 0.8`)

### Output Parameters

**`--num-samples` (default: 3)**
- Number of sample queries to show per cluster in the samples output file
- Does not affect clustering, only output

## Usage Examples

### Default Behavior
```bash
python cluster.py --input queries.csv --output-summary summary.csv
```

### Get Fewer, Larger Clusters
```bash
# Increase min-cluster-size and merge similar clusters
python cluster.py --input queries.csv \
  --min-cluster-size 10 \
  --cluster-selection-epsilon 0.3 \
  --output-summary summary.csv
```

### Get More, Smaller Clusters
```bash
# Decrease min-cluster-size and avoid merging
python cluster.py --input queries.csv \
  --min-cluster-size 2 \
  --cluster-selection-epsilon 0.0 \
  --output-summary summary.csv
```

### Get Very Tight, High-Quality Clusters
```bash
# Use when you want only the most cohesive clusters
python cluster.py --input queries.csv \
  --min-samples 5 \
  --cluster-selection-method leaf \
  --alpha 1.5 \
  --output-summary summary.csv
```

### Get Looser Clusters (More Inclusive)
```bash
# Use when you want to capture more queries in clusters
python cluster.py --input queries.csv \
  --min-samples 1 \
  --alpha 0.8 \
  --output-summary summary.csv
```

### Balanced Approach for Medium Datasets (100-1000 queries)
```bash
python cluster.py --input queries.csv \
  --min-cluster-size 5 \
  --min-samples 3 \
  --cluster-selection-epsilon 0.1 \
  --output-summary summary.csv
```

### For Large Datasets (1000+ queries)
```bash
# More conservative to avoid over-clustering
python cluster.py --input queries.csv \
  --min-cluster-size 10 \
  --min-samples 5 \
  --cluster-selection-epsilon 0.2 \
  --output-summary summary.csv
```

## Tips for Tuning

1. **Start with min-cluster-size**: This has the most direct impact on cluster count
2. **Adjust incrementally**: Change one parameter at a time to see its effect
3. **Check outliers**: If cluster -1 has many queries, your parameters may be too strict
4. **Iterate**: Clustering is exploratory; try different combinations to find what works for your data

# Visualization Options

The clustering tool supports generating 2D visualizations of your query clusters using either t-SNE or UMAP dimensionality reduction algorithms. These visualizations help you understand the cluster structure and quality.

## Generating Visualizations

### t-SNE Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) is effective for visualizing high-dimensional data by preserving local structure.

```bash
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --output-samples samples.csv \
  --visualize-tsne output/tsne_plot.png
```

**t-SNE Parameters:**

- `--tsne-perplexity` (default: 30) - Balance between local and global structure. Use 5-50 for small datasets, 30-50 for larger ones.
- `--tsne-learning-rate` (default: 200) - Learning rate for optimization. Typical range: 10-1000.
- `--tsne-max-iter` (default: 1000) - Maximum iterations for optimization. Increase if the visualization doesn't converge.

```bash
# Example with custom t-SNE parameters
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --visualize-tsne output/tsne_plot.png \
  --tsne-perplexity 15 \
  --tsne-learning-rate 150 \
  --tsne-max-iter 1500
```

### UMAP Visualization

UMAP (Uniform Manifold Approximation and Projection) is faster than t-SNE and often preserves both local and global structure better.

```bash
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --output-samples samples.csv \
  --visualize-umap output/umap_plot.png
```

**UMAP Parameters:**

- `--umap-n-neighbors` (default: 15) - Number of neighbors to consider. Smaller values focus on local structure (5-15), larger values on global structure (30-100).
- `--umap-min-dist` (default: 0.1) - Minimum distance between points in the embedding. Smaller values create tighter clusters (0.0-0.5).
- `--umap-metric` (default: 'euclidean') - Distance metric to use. Options include 'euclidean', 'manhattan', 'cosine', etc.

```bash
# Example with custom UMAP parameters
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --visualize-umap output/umap_plot.png \
  --umap-n-neighbors 10 \
  --umap-min-dist 0.05 \
  --umap-metric euclidean
```

### Generating Both Visualizations

You can generate both t-SNE and UMAP visualizations in a single run:

```bash
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --output-samples samples.csv \
  --visualize-tsne output/tsne_plot.png \
  --visualize-umap output/umap_plot.png
```

## Understanding the Visualizations

- **Colors**: Each cluster is shown in a different color
- **Black Stars** (★): Mark the centroid query of each cluster
- **X markers**: Used for outliers (cluster -1)
- **Legend**: Shows which color corresponds to each cluster number

Tight, well-separated clusters indicate good clustering quality, while overlapping clusters suggest you may need to adjust clustering parameters.

# Provenance
- https://docs.google.com/document/d/1P-r0vkVVEiCkKaIO6s2TkrO2h5Q6T5K0cxz0uMm5LM8/edit?tab=t.0
- https://chatgpt.com/share/698b6e90-06f8-800c-824c-9cabc3b926a2
- https://github.com/copilot/c/782cdd8e-fd1c-4ad2-92de-53d0dc1905d7
