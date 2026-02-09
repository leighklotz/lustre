#!/usr/bin/env python3

import argparse
import csv
import hdbscan
import numpy as np
import os
import sys
import torch
import warnings

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

# Optional: ensure you exported HF_TOKEN if you expect to use gated models.
# CodeBERT itself is public, but Hugging Face will give you an "unauthenticated" warning
# on download.
# assert os.getenv("HF_TOKEN", "") != ""

QUERY_SAMPLES = [
    # query,count,runtime,users
    ('index=web sourcetype=apache_error warn', 1434, 7, 'user1 user3'),
    ('stats count by user', 100, 5, 'user2 user4'),
    ('search index=main sourcetype=access_combined status=404', 500, 3, 'user1 user5'),
    ('index=web sourcetype=apache_error info', 200, 2, 'user3 user6'),
    ('stats count by user, host', 300, 4, 'user2 user7'),
    ('search index=main sourcetype=access_combined status=400', 600, 6, 'user4 user8'),
    ('stats count(user) by host', 150, 1, 'user5 user9'),
    ('search index=main sourcetype=access_combined status=5001', 1, 33, 'AAA BBB'),
    ('index=main sourcetype=syslog warn', 250, 5, 'user3 user11'),
    ('search index=web sourcetype=apache_access status=400', 550, 7, 'user2 user3 user12'),
    ('stats sum(bytes) as b by host, user', 350, 2, 'user4 user13'),
    ('search index=web sourcetype=apache_access status=401', 450, 4, 'user5 user14'),
    ('search index=web sourcetype=apache_access status=403', 650, 6, 'user1 user4 user15'),
    ('index=main sourcetype=syslog error', 180, 1, 'user3 user16'),
    ('search index=main sourcetype=access_combined status=401', 520, 3, 'user2 user17'),
    ('stats count(host) by user', 220, 5, 'user4 user18'),
    ('index=main sourcetype=syslog debug', 130, 2, 'user5 user19 user11'),
    ('index=web sourcetype=apache_error error', 270, 4, 'user1 user20'),
    ('stats sum(bytes) by host', 320, 6, 'user3 user21'),
    ('index=main sourcetype=syslog info', 170, 1, 'user2 user22'),
    ('search index=web sourcetype=apache_access status=500', 570, 3, 'user4 user23'),
    ('search index=web sourcetype=apache_access status=404', 620, 5, 'user5 user24'),
    ('index=web sourcetype=apache_error debug', 230, 2, 'user1 user25')
]

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
    query,count,runtime,users
    where:
        query is the query string.
        count number of times query is run
        runtime is the runtime of the query (across all times run)
        users is a string representing a list of users

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        list: A list of tuples, where each tuple contains the query string
        count, number of users, and list of users.
    """
    queries = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        assert header == ['query', 'count', 'runtime', 'users']
        for row in reader:
            query, count, runtime, users = row
            queries.append((query, int(count), int(runtime), users))
    return queries


def print_clusters(query_label_pairs, out, show_all_queries, reduced_embeddings):
    csv_writer = csv.writer(out)
    csv_headers = [ 'cluster', 'query', 'runtime', 'runcount', 'users' ]
    csv_writer.writerow(csv_headers)

    # Get the clusters in numerical order
    clusters = {}
    cluster_indices = {}  # Track original indices for embeddings
    for idx, (query, label) in enumerate(query_label_pairs):
        if label not in clusters:
            clusters[label] = []
            cluster_indices[label] = []
        clusters[label].append(query)
        cluster_indices[label].append(idx)

    if not show_all_queries:
        print_clusters_sample_query(csv_writer, clusters, cluster_indices, reduced_embeddings)
    else:
        print_clusters_all_queries(csv_writer, clusters)

# one line per query
def print_clusters_all_queries(csv_writer, clusters):
    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())
    for cluster_id in sorted_cluster_keys:
        for cluster in clusters[cluster_id]:
            (query, runtime, runcount, users) = cluster
            csv_writer.writerow([ cluster_id, query, runtime, runcount, users ])

# one line per cluster, with sample query and aggregated metrics
def print_clusters_sample_query(csv_writer, clusters, cluster_indices, reduced_embeddings):
    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())
    for cluster_id in sorted_cluster_keys:
        cluster_queries = clusters[cluster_id]
        indices = cluster_indices[cluster_id]

        # Get embeddings for this cluster
        cluster_embeddings = reduced_embeddings[indices]

        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find query closest to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        centroid_query = cluster_queries[closest_idx]

        # Aggregate metadata
        total_runtime = sum(q[1] for q in cluster_queries)
        total_runcount = sum(q[2] for q in cluster_queries)

        # Collect unique users
        all_users = set()
        for q in cluster_queries:
            users_data = q[3]
            all_users.update(users_data.split())

        # Sort users for consistent output
        sorted_users = ' '.join(sorted(all_users))

        # Output aggregated row
        csv_writer.writerow([cluster_id, centroid_query[0], total_runtime, total_runcount, sorted_users])


def main(spl_queries, show_all_queries=False):
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

    print_clusters(zip(spl_queries, cluster_labels), sys.stdout,
                   show_all_queries, reduced_embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cluster queries using CodeBERT "
                                     "and HDBSCAN. Print one line per cluster with "
                                     " one most representative query and aggregated "
                                     "metrics for the cluster.")
    parser.add_argument("--input", type=str, help="Path to the CSV file containing queries.")
    parser.add_argument("--show-all-queries", action="store_true",
                        help="Show all queries in cluster and do not aggregate metrics")
    args = parser.parse_args()

    if args.input:
        spl_queries = load_queries_from_csv(args.input)
    else:
        spl_queries = QUERY_SAMPLES

    main(spl_queries, show_all_queries=args.show_all_queries)
