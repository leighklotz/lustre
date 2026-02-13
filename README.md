# Query Cluster

**Query Cluster** groups sanitized queries into semantic similarity clusters and produces aggregate statistics and representative examples per cluster.

The tool uses **`microsoft/codebert-base`** embeddings (not configurable).
The model is downloaded automatically on first run. CUDA is used if available.

---

# Installation

Choose either CPU or GPU requirements:

```bash
git clone https://github.com/leighklotz/lustre/
cd lustre
mkdir output
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

---

# Quick Start

```bash
python cluster.py \
  --input example/example-input-queries.csv \
  --output-summary output/summary.csv \
  --output-samples output/cluster-samples.csv
```

If `--output-samples` is omitted, only the summary file is generated.

---

# Input Format

Both `.csv` and `.tsv` files are supported (delimiter inferred from extension).

## Supported Headers

The header must be one of:

```csv
query,runtime,count,users
query,runtime,users
query
```

## Field Semantics

* `query`: query string
* `runtime`: total runtime across all executions (float, e.g., `1440.2`)
* `count`: total number of executions (integer, e.g., `3`)
* `users`: space-separated list of users

If `runtime`, `count`, or `users` are omitted:

* Missing `runtime` defaults to `1`
* Missing `count` defaults to `1`
* Missing `users` defaults to empty

## Example

```csv
query,runtime,count,users
"index=web sourcetype=apache_error warn",1440.2,3,"user1 user3"
"stats count by user",100.5,2,"user2 user4"
"search index=main sourcetype=access_combined status=404",12.7,1,"user1 user5"
```

---

# Output Files

## Cluster Summary Output

One row per cluster:

* `cluster` (cluster `-1` indicates outliers)
* `cluster_size`
* `centroid_query`
* `cluster_total_runtime` (sum of runtimes in cluster)
* `cluster_total_runcount` (sum of counts in cluster)
* `cluster_all_users` (union of users, sorted)

Example:

```csv
cluster,cluster_size,centroid_query,cluster_total_runtime,cluster_total_runcount,cluster_all_users
0,4,index=web sourcetype=apache_error error,2134.7,15,user1 user3 user6
```

---

## Cluster Samples Output

Representative queries per cluster:

* `cluster`
* `cluster_size`
* `sample_type`
* `distance_from_centroid`
* `query`

`sample_type` ∈ `centroid | edge | median | diverse`

### Distance Metric

`distance_from_centroid` is **Euclidean (L2)** distance in the clustering embedding space:

* PCA-reduced space (default)
* UMAP/PCA-reduced space if `--use-umap-before-clustering` is enabled

---

# Clustering Pipeline

Default pipeline:

```
CodeBERT (768D)
→ PCA (10D)
→ Cosine similarity
→ HDBSCAN (precomputed distances)
```

Optional UMAP-based pipeline:

```
CodeBERT (768D)
→ PCA (50D)
→ UMAP (5D)
→ HDBSCAN (Euclidean metric)
```

UMAP pre-clustering:

* Applies non-linear dimensionality reduction
* Reduces dimensionality prior to density clustering
* Avoids computing a full n×n cosine distance matrix

If UMAP is not installed, PCA-only reduction is used.

---

# Clustering Parameters

### Core Parameters

* `--min-cluster-size` (default: 2)
* `--min-samples` (default: 1)
* `--cluster-selection-epsilon` (default: 0.0)
* `--cluster-selection-method` (`eom` or `leaf`)
* `--alpha` (default: 1.0)

### Output Controls

* `--output-summary`
* `--output-samples`
* `--show-all-queries`
* `--num-samples-to-show`

---

# Performance and Scaling

* Embedding generation is **O(n)** and often dominates runtime.
* Default mode computes a full **n×n cosine distance matrix (O(n²) memory and time)**.
* For larger datasets, `--use-umap-before-clustering` avoids the full distance matrix.
* Parameter values are starting points and require empirical tuning per dataset.

No specific dataset size guarantees are implied.

---

# Visualization

Optional 2D visualizations:

* `--visualize-tsne`
* `--visualize-umap`

Visualizations are generated from the same embeddings used for clustering:

* PCA 10D (default)
* UMAP/PCA 5D if UMAP pre-clustering is enabled

Markers:

* Colored points: clusters
* ★ Black star: centroid query
* X markers: outliers (`-1`)

---

# Usage Examples

### Default

```bash
python cluster.py --input queries.csv --output-summary summary.csv
```

### Fewer, Larger Clusters

```bash
python cluster.py \
  --input queries.csv \
  --min-cluster-size 10 \
  --cluster-selection-epsilon 0.3 \
  --output-summary summary.csv
```

### More, Smaller Clusters

```bash
python cluster.py \
  --input queries.csv \
  --min-cluster-size 2 \
  --cluster-selection-epsilon 0.0 \
  --output-summary summary.csv
```

### UMAP Pre-Clustering

```bash
python cluster.py \
  --input queries.csv \
  --use-umap-before-clustering \
  --output-summary summary.csv
```

---

# Provenance

* [https://docs.google.com/document/d/1P-r0vkVVEiCkKaIO6s2TkrO2h5Q6T5K0cxz0uMm5LM8/edit](https://docs.google.com/document/d/1P-r0vkVVEiCkKaIO6s2TkrO2h5Q6T5K0cxz0uMm5LM8/edit)
* [https://github.com/copilot/c/782cdd8e-fd1c-4ad2-92de-53d0dc1905d7](https://github.com/copilot/c/782cdd8e-fd1c-4ad2-92de-53d0dc1905d7)
* [https://chatgpt.com/](https://chatgpt.com/)

---

## What This Version Achieves

* No duplicated sections
* No contradictory UMAP claims
* No “5D is optimal”
* No “better clusters” promises
* Honest about O(n²)
* Explicit about float runtimes
* Explicit about default field fallbacks
* Professional, publication-level tone

If you'd like next, we can:

* Add a minimal architecture diagram,
* Add a short “Design Rationale” section (why CodeBERT + HDBSCAN),
* Or prepare this for a technical blog post or conference demo.
