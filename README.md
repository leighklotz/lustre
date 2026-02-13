# Query Cluster

This project clusters sanitized queries into similarity groups and gives aggregate statistics and one or more sample queries.
The tool currently uses **microsoft/codebert-base** embeddings (not configurable).
The model is downloaded automatically on first run. CUDA is used automatically if available.

---

# Installation and Quick Start

1. First, choose either `requirements.txt` or `requirements-gpu.txt`.

```bash
$ git clone https://github.com/leighklotz/lustre/
$ cd lustre
$ mkdir output
$ python --version
Python 3.12.3
$ python3 -m venv .venv
$ . .venv/bin/activate
(.venv) $ pip install -r requirements.txt 
```

Below is a simple run. For a more complex run, see [example](example).

```bash
(.venv) $ python cluster.py --input example/example-input-queries.csv --output-summary output/summary.csv --output-samples output/cluster-samples.csv
```

Note: The `--output-samples` parameter is optional. If omitted, only the summary file will be generated.

---

## Input `queries.csv` and `queries.tsv` format:

The header must one of the following:

```csv
query,runtime,count,users
query,runtime,users
query
```

* `runtime`: total runtime across all executions of the query (integer seconds)
* `count`: total number of times the query was run (integer)
* `users`: space-separated list of users who ran the query
* `.csv` and `.tsv` files are supported (delimiter inferred from extension).

### Example: 

```csv
query,runtime,count,users
"index=web sourcetype=apache_error warn",1434,7,"user1 user3"
"stats count by user",100,5,"user2 user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1 user5"
```
## Summary File Output fields
* `cluster` (`cluster` is -1 for outlier cluster)
* `cluster_size`
* `centroid_query`
* `cluster_total_runtime`
* `cluster_total_runcount`
* `cluster_all_users`

---

## Samples File Output Fields

* `cluster`
* `cluster_size`
* `sample_type`
* `distance_from_centroid`
* `query`

(`sample_type` is one of `median` | `edge` | `centroid` | `diverse`)

--- 
### Clustering Embedding Distance Metric

`distance_from_centroid` is Euclidean (L2) distance in the clustering embedding space:
* PCA-reduced space by default
* UMAP/PCA-reduced space if `--use-umap-before-clustering` is enabled

---

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

---

## Cluster Samples Output

Below are the selected queries from each cluster. There are three sample queries per cluster.

When you specify `--output-samples <filepath>` it outputs some sample queries from each cluster to a separate file, so you can inspect the queries extracted to each cluster more closely.

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

---

# Cluster Tuning Parameters

## Controlling Cluster Size and Tightness

The following parameters control how HDBSCAN clusters your queries. Use these to adjust cluster size, tightness, and the number of clusters produced.

### Core Parameters


### Output Parameters

**`--output-summary` (default: stdout)**
* Path to output CSV file for cluster summary
* If not specified, outputs to stdout

**`--output-samples`**
* Path to output CSV file for sample queries
* Optional: if not specified, no samples file is generated

**`--show-all-queries`**
* When enabled, outputs all queries in each cluster without aggregating metrics
* Produces a single file output instead of separate summary and samples files
* Useful for detailed cluster inspection

**`--num-samples-to-show` (default: 3)**
* Number of sample queries to show per cluster in the samples output file
* Does not affect clustering, only output

**`--min-cluster-size` (default: 2)**
Minimum number of queries required to form a cluster.

**`--min-samples` (default: 1)**
Number of samples in a neighborhood for a point to be considered a core point.

**`--cluster-selection-epsilon` (default: 0.0)**
Distance threshold influencing post-processing merging of nearby clusters.

**`--cluster-selection-method` (default: 'eom')**
Options: `eom` or `leaf`.

**`--alpha` (default: 1.0)**
Conservativeness parameter for cluster selection.

---

# Advanced: UMAP Pre-Clustering

By default, the clustering tool uses PCA (10D) followed by cosine similarity and HDBSCAN with precomputed distances.

When enabled, the pipeline changes from:

```
CodeBERT Embeddings (768D) → PCA (10D) → Cosine Distance → HDBSCAN
```

To:

```
CodeBERT Embeddings (768D) → PCA (50D) → UMAP (5D) → HDBSCAN (Euclidean)
```

UMAP pre-clustering:

* Applies non-linear manifold learning
* Uses 5D as a common target dimensionality for density clustering
* May improve cluster separation for complex query corpora
* Avoids computing a full n×n cosine distance matrix

If UMAP is not installed, the tool falls back to PCA-only reduction.

---

# Performance and Scaling Notes

* Embedding generation is O(n) and typically dominates runtime for moderate datasets.
* Default mode computes a full n×n cosine distance matrix (O(n²) memory and time), which limits practical dataset size.
* For larger datasets, prefer `--use-umap-before-clustering` to avoid the full n×n distance matrix.
* Parameter suggestions are starting points and may require empirical tuning for your dataset.

---

# Visualization Options

Visualization options (`--visualize-tsne` and `--visualize-umap`) are separate from UMAP pre-clustering.

Visualizations are generated from the same embeddings used for clustering:

* PCA 10D by default
* UMAP/PCA 5D if `--use-umap-before-clustering` is enabled

Colors represent clusters.
Black stars (★) mark centroid queries.
X markers represent outliers (cluster -1).

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

## Tips for Tuning

1. **Start with min-cluster-size**: This has the most direct impact on cluster count
2. **Adjust incrementally**: Change one parameter at a time to see its effect
3. **Check outliers**: If cluster -1 has many queries, your parameters may be too strict
4. **Iterate**: Clustering is exploratory; try different combinations to find what works for your data

# Advanced: UMAP Pre-Clustering

By default, the clustering tool uses PCA (10D) followed by cosine similarity and HDBSCAN with precomputed distances. For improved cluster quality and separation, you can enable optional UMAP-based dimensionality reduction before clustering.

## Why Use UMAP Pre-Clustering?

The default approach has some limitations:
* **PCA is linear** and may miss non-linear semantic patterns in embeddings
* **10D is relatively high** for density-based clustering (curse of dimensionality)
* **Precomputed distances** are less flexible than direct metric computation

UMAP pre-clustering addresses these issues:
* **Non-linear manifold learning** preserves semantic structure better
* **5D is optimal** for HDBSCAN density calculations
* **Better cluster separation** and fewer outliers
* **Well-established approach** in NLP/ML communities

## How It Works

When enabled, the pipeline changes from:
```
CodeBERT Embeddings (768D) → PCA (10D) → Cosine Distance → HDBSCAN
```

To:
```
CodeBERT Embeddings (768D) → PCA (50D) → UMAP (5D) → HDBSCAN (Euclidean)
```

The 50D PCA step before UMAP helps denoise and speeds up UMAP computation.

## Usage

### Basic Usage

Enable UMAP pre-clustering with the `--use-umap-before-clustering` flag:

```bash
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --output-samples samples.csv \
  --use-umap-before-clustering
```

### Custom Parameters

Fine-tune the UMAP reduction with these parameters:

* `--umap-cluster-n-components` (default: 5) - Target dimensionality for clustering. 5D is optimal for HDBSCAN.
* `--umap-cluster-n-neighbors` (default: 20) - Balance between local and global structure (15-30 recommended).
* `--umap-cluster-min-dist` (default: 0.0) - Minimum distance between points. Use 0.0 for maximum cluster separation.
* `--umap-cluster-metric` (default: 'cosine') - Distance metric. 'cosine' is best for text embeddings.

```bash
# Example with custom UMAP pre-clustering parameters
python cluster.py --input queries.csv \
  --output-summary summary.csv \
  --use-umap-before-clustering \
  --umap-cluster-n-neighbors 30 \
  --umap-cluster-min-dist 0.0 \
  --umap-cluster-metric cosine
```

## Expected Improvements

When using UMAP pre-clustering, you can expect:
* **Better cluster separation** - More distinct, well-separated clusters
* **Fewer outliers** - Reduced number of queries in cluster -1
* **More semantic coherence** - Clusters better reflect semantic similarity
* **Faster HDBSCAN** - 5D vs 10D reduces computational cost

## Fallback Behavior

If UMAP is not installed, the tool automatically falls back to a PCA-only approach:
* PCA is reduced to 5D (instead of UMAP)
* You'll see a warning message
* Install umap-learn to use the full UMAP approach: `pip install umap-learn`

## When to Use

Consider using UMAP pre-clustering when:
* You have complex, semantically diverse queries
* Default clustering produces too many outliers
* You want better cluster quality and separation
* You have a larger dataset (100+ queries)

## Comparison with Default Approach

| Aspect | Default (PCA + Cosine) | UMAP Pre-Clustering |
|--------|------------------------|---------------------|
| Dimensionality Reduction | Linear (PCA 10D) | Non-linear (PCA 50D → UMAP 5D) |
| Clustering Metric | Precomputed cosine distance | Euclidean distance |
| Cluster Quality | Good for linear patterns | Better for complex patterns |
| Outlier Rate | May be higher | Usually lower |
| Computation Time | Faster | Slightly slower (UMAP overhead) |
| Best For | Quick analysis, simple patterns | Complex queries, better quality |

## Note

UMAP pre-clustering is an **optional feature** that is **off by default**. The original PCA + precomputed distance approach remains the default to maintain backward compatibility.

# Visualization Options

The clustering tool supports generating 2D visualizations of your query clusters using either t-SNE or UMAP dimensionality reduction algorithms. These visualizations help you understand the cluster structure and quality.

**Note:** The visualization options (`--visualize-tsne` and `--visualize-umap`) are separate from the UMAP pre-clustering feature. Visualization creates 2D plots, while UMAP pre-clustering improves the actual clustering algorithm.

**Important:** UMAP parameters have different names depending on their purpose:
* **Visualization parameters** use `--umap-*` (e.g., `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric`)
* **Pre-clustering parameters** use `--umap-cluster-*` (e.g., `--umap-cluster-n-neighbors`, `--umap-cluster-min-dist`, `--umap-cluster-metric`)

These are separate features and their parameters do not affect each other.

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

* `--tsne-perplexity` (default: 30) - Balance between local and global structure. Use 5-50 for small datasets, 30-50 for larger ones.
* `--tsne-learning-rate` (default: 200) - Learning rate for optimization. Typical range: 10-1000.
* `--tsne-max-iter` (default: 1000) - Maximum iterations for optimization. Increase if the visualization doesn't converge.

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

* `--umap-n-neighbors` (default: 15) - Number of neighbors to consider. Smaller values focus on local structure (5-15), larger values on global structure (30-100).
* `--umap-min-dist` (default: 0.1) - Minimum distance between points in the embedding. Smaller values create tighter clusters (0.0-0.5).
* `--umap-metric` (default: 'euclidean') - Distance metric to use. Options include 'euclidean', 'manhattan', 'cosine', etc.

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

* **Colors**: Each cluster is shown in a different color
* **Black Stars** (★): Mark the centroid query of each cluster
* **X markers**: Used for outliers (cluster -1)
* **Legend**: Shows which color corresponds to each cluster number

Tight, well-separated clusters indicate good clustering quality, while overlapping clusters suggest you may need to adjust clustering parameters.

# Provenance
* https://docs.google.com/document/d/1P-r0vkVVEiCkKaIO6s2TkrO2h5Q6T5K0cxz0uMm5LM8/edit?tab=t.0
* https://chatgpt.com/share/698b6e90-06f8-800c-824c-9cabc3b926a2
* https://github.com/copilot/c/782cdd8e-fd1c-4ad2-92de-53d0dc1905d7
* https://chatgpt.com/c/698e80a1-2d30-8327-b3f6-c91578cd3c74
