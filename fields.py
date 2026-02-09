#!/usr/bin/env python3

import os
import sys
import numpy as np
import hdbscan
import torch
import csv

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Optional: ensure you exported HF_TOKEN if you expect to use gated models.
# CodeBERT itself is public, but Hugging Face will give you an "unauthenticated" warning
# on download.
# assert os.getenv("HF_TOKEN", "") != ""

SPL_QUERY_SAMPLES = [
    ('index=web sourcetype=apache_error warn', 1434, 7, ['user1', 'user3']),
    ('stats count by user', 100, 5, ['user2', 'user4']),
    ('search index=main sourcetype=access_combined status=404', 500, 3, ['user1', 'user5']),
    ('index=web sourcetype=apache_error info', 200, 2, ['user3', 'user6']),
    ('stats count by user, host', 300, 4, ['user2', 'user7']),
    ('search index=main sourcetype=access_combined status=400', 600, 6, ['user4', 'user8']),
    ('stats count(user) by host', 150, 1, ['user5', 'user9']),
    ('search index=main sourcetype=access_combined status=5001', 1, 33, ['AAA', 'BBB']),
    ('index=main sourcetype=syslog warn', 250, 5, ['user3', 'user11']),
    ('search index=web sourcetype=apache_access status=400', 550, 7, ['user2', 'user12']),
    ('stats sum(bytes) as b by host, user', 350, 2, ['user4', 'user13']),
    ('search index=web sourcetype=apache_access status=401', 450, 4, ['user5', 'user14']),
    ('search index=web sourcetype=apache_access status=403', 650, 6, ['user1', 'user15']),
    ('index=main sourcetype=syslog error', 180, 1, ['user3', 'user16']),
    ('search index=main sourcetype=access_combined status=401', 520, 3, ['user2', 'user17']),
    ('stats count(host) by user', 220, 5, ['user4', 'user18']),
    ('index=main sourcetype=syslog debug', 130, 2, ['user5', 'user19']),
    ('index=web sourcetype=apache_error error', 270, 4, ['user1', 'user20']),
    ('stats sum(bytes) by host', 320, 6, ['user3', 'user21']),
    ('index=main sourcetype=syslog info', 170, 1, ['user2', 'user22']),
    ('search index=web sourcetype=apache_access status=500', 570, 3, ['user4', 'user23']),
    ('search index=web sourcetype=apache_access status=404', 620, 5, ['user5', 'user24']),
    ('index=web sourcetype=apache_error debug', 230, 2, ['user1', 'user25'])
]

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

def load_queries_from_csv(csv_file):
    """
    Loads queries from a CSV file.

    The CSV file should have the following format:
    query,count,num_users,users
    where:
        query is the SPL query string.
        count is a numerical value (not used in clustering, but kept for consistency of original dataset).
        num_users is a numerical value (not used in clustering, but kept for consistency of original dataset).
        users is a string representing a list of users (not used in clustering).

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        list: A list of tuples, where each tuple contains the query string, count, number of users, and list of users.
    """
    queries = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            query, count, num_users, users = row
            queries.append((query, int(count), int(num_users), users))
    return queries


def print_clusters(query_label_pairs, out):
    csv_writer = csv.writer(out)
    csv_headers = [ 'cluster', 'query', 'runtime', 'runcount', 'users' ]
    csv_writer.writerow(csv_headers) 

    # Get the clusters in numerical order
    clusters = {}
    for query, label in query_label_pairs:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(query)

    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())

    for cluster_id in sorted_cluster_keys:
        for cluster in clusters[cluster_id]:
            (query, runtime, runcount, users) = cluster
            users = ' '.join(users)
            csv_writer.writerow([ cluster_id, query, runtime, runcount, users ])

def main(spl_queries):
    # build embedding matrix: (n_samples, hidden)
    embeddings = np.vstack([get_embedding(q[0]) for q in spl_queries]).astype(np.float64)

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

    print_clusters(zip(spl_queries, cluster_labels), sys.stdout)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster SPL queries using CodeBERT and HDBSCAN.")
    parser.add_argument("--input", type=str, help="Path to the CSV file containing SPL queries.")
    args = parser.parse_args()
    
    if args.input:
        spl_queries = load_queries_from_csv(args.input)
    else:
        # Example Usage (Replace With Your Spl Queries)
        spl_queries = SPL_QUERY_SAMPLES

    main(spl_queries)

