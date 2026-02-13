# Worked Example: Recovering Saved Query Templates

**Goal:**

Given a large corpus of realized queries from audit logs,
cluster them into groups corresponding to original saved searches
or dashboard panels.

Each realized query is assumed to be:

    base_template + small perturbation

Perturbations include:
- Field value changes (status=404 → status=500)
- Optional stanzas
- Minor field renaming
- Small clause variations

---

## 1. Problem Framing

We want clusters to represent structural templates,
not literal token identity.

Desired cluster properties:

* High structural similarity
* Low angular variance
* Small edit distance
* Same logical skeleton

Statistically:

Minimize intra-cluster cosine variance
while preserving template support.

---

## 2. Recommended Configuration

For template recovery:

```bash
--use-umap-before-clustering
--min-cluster-size 5–20
--min-samples 3–5
--cluster-selection-epsilon 0.1–0.3
```

Interpretation:

* Require multiple realizations of a template
* Allow small structural perturbations
* Merge nearby density regions

---

## 3. Tuning Strategy

### Phase 1 – Remove Rare Queries

Increase `min-cluster-size` until:

* Most single-use queries become outliers
* Clusters represent recurring structures

---

### Phase 2 – Merge Variants

If clusters split by field values:

Example:

```
status=404
status=500
status=403
```

Increase:

```
--cluster-selection-epsilon
```

This merges close density islands.

---

### Phase 3 – Enforce Structural Cohesion

If unrelated templates merge:

Increase:

```
--min-samples
```

This requires stronger local density.

---

## 4. Statistical Validation

For each cluster, compute:

### Intra-Cluster Cosine Variance

Low variance → strong template identity.

---

### Token Overlap Score

Compute Jaccard similarity of token sets.
Template clusters show high overlap.

---

### Structural Stability

Re-run clustering with small parameter perturbations.

Stable clusters should:

* Persist across runs
* Maintain similar membership

---

## 5. Large Live Corpus Workflow

1. Run clustering on full audit set
2. Inspect cluster size histogram
3. Identify top N clusters
4. Inspect centroids
5. Compare with saved search definitions
6. Adjust epsilon and min-samples
7. Repeat

Treat this as iterative model fitting.

---

## 6. Failure Modes

| Issue                             | Likely Cause             | Fix                  |
| --------------------------------- | ------------------------ | -------------------- |
| Template split into many clusters | epsilon too small        | Increase epsilon     |
| Mixed templates in cluster        | density too loose        | Increase min-samples |
| Many singleton clusters           | min-cluster-size too low | Increase threshold   |

---

## 7. Interpretation

If tuning is successful:

Each cluster approximates:

```
structural equivalence class of a saved query
```

The centroid approximates:

```
canonical template realization
```

Clusters represent:

```
isomorphic query families
```
