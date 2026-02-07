#!/usr/bin/env python3
import os
import numpy as np
import hdbscan
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Optional: ensure you exported HF_TOKEN if you expect to use gated models.
# CodeBERT itself is public, so this is not strictly required.
# assert os.getenv("HF_TOKEN", "") != ""

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

def get_embedding(text: str) -> np.ndarray:
    """Return a single (hidden_size,) embedding vector for `text`."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over sequence length -> (1, hidden)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb[0]  # -> (hidden,)

# Example usage (replace with your SPL queries)
spl_queries = [
    "index=web sourcetype=apache_error WARN",
    "stats count by user",
    "search index=main sourcetype=access_combined status=404",
    "index=web sourcetype=apache_error INFO",
    "stats count by user, host",
    "search index=main sourcetype=access_combined status=400",
    "stats count(user) by host",
    "search index=main sourcetype=access_combined status=500",
    "index=main sourcetype=syslog WARN",
    "search index=web sourcetype=apache_access status=400",
    "stats sum(bytes) as b by host, user",
    "search index=web sourcetype=apache_access status=401",
    "search index=web sourcetype=apache_access status=403",
    "index=main sourcetype=syslog ERROR",
    "search index=main sourcetype=access_combined status=401",
    "stats count(host) by user",
    "index=main sourcetype=syslog DEBUG",
    "index=web sourcetype=apache_error ERROR",
    "stats sum(bytes) by host",
    "index=main sourcetype=syslog INFO",
    "search index=web sourcetype=apache_access status=500",
    "search index=web sourcetype=apache_access status=404",
    "index=web sourcetype=apache_error DEBUG"
]

# Build embedding matrix: (n_samples, hidden)
embeddings = np.vstack([get_embedding(q) for q in spl_queries]).astype(np.float64)

# Dimensionality reduction (PCA)
pca = PCA(n_components=10, random_state=0)
reduced_embeddings = pca.fit_transform(embeddings)

# HDBSCAN expects a DISTANCE matrix when metric='precomputed'.
# You were passing a SIMILARITY matrix; convert to distance.
S = cosine_similarity(reduced_embeddings)          # similarity in [-1, 1]
D = 1.0 - S                                       # cosine distance in [0, 2]
np.fill_diagonal(D, 0.0)

clusterer = hdbscan.HDBSCAN(
    metric="precomputed",
    min_cluster_size=2,   # tiny dataset => smaller clusters allowed
    min_samples=1,
    cluster_selection_epsilon=0.0,
)

cluster_labels = clusterer.fit_predict(D)

# TODO: Print the clusters in numerical order
clusters = {}
for query, label in zip(spl_queries, cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(query)

for label, queries in clusters.items():
    print(f"Cluster {label}:")
    for query in queries:
        print(f"- {query}")

