# Query Cluster – Tuning Guide

This document explains how to tune clustering behavior for different analytical goals.

Assumptions:
- Reader understands embeddings, PCA/UMAP, density clustering.
- Reader may use an LLM to assist in iterative refinement.

Clustering is exploratory model selection. There is no universal parameter set.

---

## 1. Pipeline Overview

Default pipeline:

```
CodeBERT (768D)
→ PCA (10D)
→ cosine distance
→ HDBSCAN (precomputed)
```

Optional UMAP mode:

```
CodeBERT
→ PCA (50D)
→ UMAP (5D)
→ HDBSCAN (Euclidean)
```

Tuning affects:

* Geometry (PCA depth, UMAP settings)
* Density thresholds (HDBSCAN parameters)
* Cluster granularity

---

## 2. Tuning Workflow

### Step 1 – Examine Outliers

Cluster `-1` size indicates strictness.

* Too many outliers → increase permissiveness
* Too few → clusters may be overly loose

---

### Step 2 – Examine Cluster Size Distribution

Inspect:

* Median cluster size
* Number of size-2 clusters
* Heavy tail behavior

Interpretation:

* Many tiny clusters → too permissive
* One dominant cluster → too loose

---

### Step 3 – Inspect Samples

Use:

```
--output-samples
```

Verify:

* Centroid represents cluster theme
* Edge queries are structurally related
* No obvious cross-template mixing

---

### Step 4 – Adjust Density Parameters

#### `--min-cluster-size`

Primary granularity control.

* Increase → fewer, larger clusters
* Decrease → more, smaller clusters

Interpretation: minimum template support.

---

#### `--min-samples`

Controls density conservativeness.

* Increase → tighter cores, more outliers
* Decrease → looser density

Interpretation: required local agreement.

---

#### `--cluster-selection-epsilon`

Merges nearby density regions.

* Increase → unify similar clusters
* Decrease → preserve distinctions

Use when minor variations split clusters.

---

## 3. When to Use UMAP Mode

Use `--use-umap-before-clustering` when:

* Dataset is large
* Default PCA produces fragmentation
* Embedding manifold is non-linear

UMAP reduces dimensionality and avoids O(n²) distance computation.

---

## 4. Scaling Notes

* Embedding generation: O(n)
* Default clustering: O(n²) memory/time
* UMAP mode: avoids full distance matrix

No dataset size guarantees are implied.

---

## 5. Diagnostic Patterns

| Symptom                        | Likely Cause      | Adjustment                         |
| ------------------------------ | ----------------- | ---------------------------------- |
| Many size-2 clusters           | Too permissive    | Increase min-cluster-size          |
| Template split across clusters | Epsilon too small | Increase cluster-selection-epsilon |
| One giant cluster              | Too loose         | Increase min-samples               |
| High outlier rate              | Too strict        | Reduce min-samples                 |

---

## 6. Optional Geometry Refinement

Advanced users may experiment with:

* L2-normalizing embeddings before PCA
* Increasing PCA dimensions (10 → 20)
* Adjusting UMAP `n_neighbors` and `min_dist`

These refine topology but do not fundamentally change clustering logic.

---

## 7. Philosophy

Treat clustering parameters as hyperparameters.

Iterate:

1. Run clustering
2. Inspect histogram
3. Inspect samples
4. Adjust parameters
5. Repeat

Clustering quality is determined by alignment between:

* Embedding geometry
* Density thresholds
* Analytical goal
