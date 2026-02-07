#!/usr/bine/env python3
import os
import numpy as np
import hdbscan
import torch

# see .env for export
assert os.getenv('HF_TOKEN', '') != ''

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get a fixed-size embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Example usage (replace with your SPL queries)
spl_queries = [
    "search index=main sourcetype=access_combined status=404",
    "search index=main sourcetype=access_combined status=400",
    "search index=main sourcetype=access_combined status=401",
    "search index=main sourcetype=access_combined status=500",
    "search index=web sourcetype=apache_access status=500",
    "search index=web sourcetype=apache_access status=400",
    "search index=web sourcetype=apache_access status=401",
    "search index=web sourcetype=apache_access status=404",
    "search index=web sourcetype=apache_access status=403",
    "stats count by user",
    "stats count by user, host",
    "stats count(host) by user",
    "stats count(user) by host",
    "stats sum(bytes) by host",
    "stats sum(bytes) as b by host, user",
    "index=main sourcetype=syslog ERROR",
    "index=main sourcetype=syslog WARN",
    "index=main sourcetype=syslog INFO",
    "index=main sourcetype=syslog DEBUG",
    "index=web sourcetype=apache_error ERROR",
    "index=web sourcetype=apache_error WARN",
    "index=web sourcetype=apache_error INFO",
    "index=web sourcetype=apache_error DEBUG"
]

embeddings = [get_embedding(query) for query in spl_queries]
embeddings = np.concatenate(embeddings) # Combine into a single array

# Dimensionality Reduction (PCA)
pca = PCA(n_components=10)
reduced_embeddings = pca.fit_transform(embeddings)

# HDBSCAN Clustering
# Pass the precomputed distance matrix to hdbscan
cosine_sim = cosine_similarity(reduced_embeddings).astype(np.float64)
# Adjust epsilon value
clusterer = hdbscan.HDBSCAN(metric='cosine', cluster_selection_epsilon=0.0)
cluster_labels = clusterer.fit_predict(cosine_sim)

# Print results
for i, label in enumerate(cluster_labels):
    print(f"Query: {spl_queries[i//1]} Cluster: {label}") #Corrected indexing
