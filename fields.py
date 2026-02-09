#!/usr/bin/env python3

import os
import sys
import numpy as np
import hdbscan
import torch
import csv
import warnings

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


def normalize_users(users_data):
    """
    Normalize users data to space-separated string format.
    
    Args:
        users_data: Either a list of users or a comma-separated string
    
    Returns:
        str: Space-separated string of users
    """
    if isinstance(users_data, list):
        return ' '.join(users_data)
    elif isinstance(users_data, str):
        # Convert comma-separated to space-separated
        if ',' in users_data:
            return ' '.join(u.strip() for u in users_data.split(',') if u.strip())
        return users_data
    return str(users_data)


def print_clusters(query_label_pairs, out, aggregate=False, reduced_embeddings=None):
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

    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())

    if aggregate and reduced_embeddings is None:
        # Warn if aggregate is requested but embeddings are not available
        warnings.warn("Aggregate mode requested but reduced_embeddings not provided. Falling back to default mode.", UserWarning)

    if aggregate and reduced_embeddings is not None:
        # Aggregate mode: one line per cluster
        for cluster_id in sorted_cluster_keys:
            cluster_queries = clusters[cluster_id]
            indices = cluster_indices[cluster_id]
            
            # Get embeddings for this cluster
            cluster_embeddings = reduced_embeddings[indices]
            
            # Calculate centroid
            centroid = cluster_embeddings.mean(axis=0)
            
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
                if isinstance(users_data, list):
                    all_users.update(users_data)
                elif isinstance(users_data, str):
                    # Handle comma-separated string
                    all_users.update(u.strip() for u in users_data.split(',') if u.strip())
            
            # Sort users for consistent output
            sorted_users = ' '.join(sorted(all_users))
            
            # Output aggregated row
            csv_writer.writerow([cluster_id, centroid_query[0], total_runtime, total_runcount, sorted_users])
    else:
        # Default mode: all queries
        for cluster_id in sorted_cluster_keys:
            for cluster in clusters[cluster_id]:
                (query, runtime, runcount, users) = cluster
                # Normalize users to space-separated format
                users = normalize_users(users)
                csv_writer.writerow([ cluster_id, query, runtime, runcount, users ])

def main(spl_queries, aggregate=False):
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
                   aggregate=aggregate, reduced_embeddings=reduced_embeddings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster SPL queries using CodeBERT and HDBSCAN.")
    parser.add_argument("--input", type=str, help="Path to the CSV file containing SPL queries.")
    parser.add_argument("--aggregate", action="store_true", 
                        help="Print one line per cluster with aggregated metadata and centroid query.")
    args = parser.parse_args()
    
    if args.input:
        spl_queries = load_queries_from_csv(args.input)
    else:
        # Example Usage (Replace With Your Spl Queries)
        spl_queries = SPL_QUERY_SAMPLES

    main(spl_queries, aggregate=args.aggregate)

