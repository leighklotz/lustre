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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

## TODO: Make MIN_CLUSTER_SIZE a default and add a CLI parameter override

# tiny dataset => smaller clusters allowed
MIN_CLUSTER_SIZE=2

# Optional: ensure you exported HF_TOKEN if you expect to use gated models.
# CodeBERT itself is public, but Hugging Face will give you an "unauthenticated" warning
# on download.
# assert os.getenv("HF_TOKEN", "") != ""

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
        runtime is the runtime of the query (across all times run)
        count number of times query is run
        users is a string representing a space-separated list of users

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        list: A list of tuples, where each tuple contains the query string
        count, number of users, and space-separated list of users.
    """
    queries = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        assert header == ['query', 'runtime', 'count', 'users']
        for row in reader:
            query, runtime, count, users = row
            queries.append((query, int(runtime), int(count), users))
    return queries


# Get the clusters in numerical order: clusters
# Track original indices for embeddings: cluster_indices
def get_clusters(query_label_pairs):
    clusters = {}
    cluster_indices = {}
    for idx, (query, label) in enumerate(query_label_pairs):
        if label not in clusters:
            clusters[label] = []
            cluster_indices[label] = []
        clusters[label].append(query)
        cluster_indices[label].append(idx)

    return clusters, cluster_indices


def get_sample_queries(cluster_embeddings, cluster_queries, num_samples=3):
    """
    Select representative queries from a cluster.
    Returns: list of (query_tuple, role, distance) tuples where:
        - query_tuple is (query, runtime, runcount, users) and users is a space-separated string
        - role is 'centroid', 'edge', 'median', or 'diverse'
        - distance is the distance from the centroid
    """
    samples = []
    
    # Calculate centroid
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    
    # 1. Centroid query (closest to center)
    centroid_idx = np.argmin(distances)
    samples.append((cluster_queries[centroid_idx], 'centroid', distances[centroid_idx]))
    
    if num_samples > 1 and len(cluster_queries) > 1:
        # 2. Edge query (farthest from center)
        edge_idx = np.argmax(distances)
        if edge_idx != centroid_idx:
            samples.append((cluster_queries[edge_idx], 'edge', distances[edge_idx]))
    
    if num_samples > 2 and len(cluster_queries) > 2:
        # 3. Median query (medium distance)
        median_idx = np.argsort(distances)[len(distances) // 2]
        if median_idx not in [centroid_idx, edge_idx]:
            samples.append((cluster_queries[median_idx], 'median', distances[median_idx]))
    
    # If we need more samples, use diversity sampling
    if num_samples > 3 and len(cluster_queries) > 3:
        selected_indices = {centroid_idx, edge_idx, median_idx}
        while len(samples) < num_samples and len(selected_indices) < len(cluster_queries):
            # Find query farthest from all selected queries
            max_min_dist = -1
            best_idx = -1
            for i in range(len(cluster_queries)):
                if i in selected_indices:
                    continue
                # Min distance to any selected query
                min_dist = min(np.linalg.norm(cluster_embeddings[i] - cluster_embeddings[j]) 
                              for j in selected_indices)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            if best_idx >= 0:
                samples.append((cluster_queries[best_idx], 'diverse', distances[best_idx]))
                selected_indices.add(best_idx)
    
    return samples


def print_clusters(query_label_pairs, summary_out, samples_out, show_all_queries, reduced_embeddings, num_samples=3):
    clusters, cluster_indices = get_clusters(query_label_pairs)

    if not show_all_queries:
        print_clusters_two_files(clusters, cluster_indices, reduced_embeddings, 
                                summary_out, samples_out, num_samples)
    else:
        print_clusters_all_queries(summary_out, clusters)


# one line per query - original behavior
def print_clusters_all_queries(out, clusters):
    csv_writer = csv.writer(out)
    csv_headers = ['cluster', 'cluster_size', 'query', 'runtime', 'runcount', 'users']
    csv_writer.writerow(csv_headers)
    
    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())
    for cluster_id in sorted_cluster_keys:
        cluster_size = len(clusters[cluster_id])
        for cluster in clusters[cluster_id]:
            (query, runtime, runcount, users) = cluster
            csv_writer.writerow([cluster_id, cluster_size, query, runtime, runcount, users])


# Two-file output:
# File 1 (summary): One row per cluster with centroid query and cluster stats
# File 2 (samples): N rows per cluster with sample queries (no stats)
def print_clusters_two_files(clusters, cluster_indices, reduced_embeddings, 
                             summary_out, samples_out, num_samples=3):
    summary_writer = csv.writer(summary_out)
    
    # Summary file headers
    summary_headers = ['cluster', 'cluster_size', 'centroid_query', 
                      'cluster_total_runtime', 'cluster_total_runcount', 'cluster_all_users']
    summary_writer.writerow(summary_headers)
    
    # Samples file (only if output file specified)
    if samples_out:
        samples_writer = csv.writer(samples_out)
        samples_headers = ['cluster', 'cluster_size', 'sample_type', 'distance_from_centroid', 'query']
        samples_writer.writerow(samples_headers)
    
    # Sort cluster keys to print in numerical order
    sorted_cluster_keys = sorted(clusters.keys())
    for cluster_id in sorted_cluster_keys:
        cluster_queries = clusters[cluster_id]
        indices = cluster_indices[cluster_id]
        cluster_size = len(cluster_queries)

        # Get embeddings for this cluster
        cluster_embeddings = reduced_embeddings[indices]

        # Get multiple sample queries
        samples = get_sample_queries(cluster_embeddings, cluster_queries, num_samples)
        
        # Aggregate metadata for entire cluster
        total_runtime = sum(q[1] for q in cluster_queries)
        total_runcount = sum(q[2] for q in cluster_queries)
        
        # Collect unique users
        all_users = set()
        for q in cluster_queries:
            users_data = q[3]
            all_users.update(users_data.split()) # space-separated

        # Sort users for consistent output
        sorted_users = ' '.join(sorted(all_users))
        
        # Write summary file (one row per cluster with centroid)
        centroid_query = samples[0][0][0]  # First sample is centroid, get query text
        summary_writer.writerow([
            cluster_id,
            cluster_size,
            centroid_query,
            total_runtime,
            total_runcount,
            sorted_users
        ])
        
        # Write samples file (N rows per cluster) - only query text, no stats
        if samples_out:
            for query_tuple, sample_type, distance in samples:
                samples_writer.writerow([
                    cluster_id,
                    cluster_size,
                    sample_type,
                    f"{distance:.4f}",
                    query_tuple[0]      # query text only
                ])


def visualize_clusters(reduced_embeddings, cluster_labels, clusters, cluster_indices, 
                       output_path, method='tsne', **method_params):
    """
    Visualize clusters using dimensionality reduction (t-SNE or UMAP).
    
    Args:
        reduced_embeddings: The PCA-reduced embeddings (from main function)
        cluster_labels: Array of cluster labels for each query
        clusters: Dict mapping cluster_id -> list of query tuples
        cluster_indices: Dict mapping cluster_id -> list of original indices
        output_path: Path to save the visualization
        method: 'tsne' or 'umap'
        **method_params: Additional parameters for the reduction method
    """
    # Apply dimensionality reduction to 2D
    if method == 'tsne':
        # Default t-SNE parameters
        perplexity = method_params.get('perplexity', 30)
        learning_rate = method_params.get('learning_rate', 200)
        n_iter = method_params.get('n_iter', 1000)
        random_state = method_params.get('random_state', 42)
        
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                      learning_rate=learning_rate, n_iter=n_iter, 
                      random_state=random_state)
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            print("Error: umap-learn package not installed. Please install with: pip install umap-learn")
            return
        
        # Default UMAP parameters
        n_neighbors = method_params.get('n_neighbors', 15)
        min_dist = method_params.get('min_dist', 0.1)
        metric = method_params.get('metric', 'euclidean')
        random_state = method_params.get('random_state', 42)
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                           min_dist=min_dist, metric=metric,
                           random_state=random_state)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # Reduce to 2D
    embeddings_2d = reducer.fit_transform(reduced_embeddings)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get unique cluster labels and sort them
    unique_labels = sorted(set(cluster_labels))
    
    # Create a colormap - use a colorblind-friendly palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each cluster
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_points = embeddings_2d[mask]
        
        # Use different markers for outliers vs regular clusters
        if label == -1:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=[color_map[label]], label=f'Outliers (cluster {label})',
                       alpha=0.6, s=50, marker='x')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=[color_map[label]], label=f'Cluster {label}',
                       alpha=0.6, s=50)
    
    # Mark centroids with distinct markers
    for cluster_id in unique_labels:
        if cluster_id in cluster_indices:
            indices = cluster_indices[cluster_id]
            cluster_embeddings = reduced_embeddings[indices]
            
            # Calculate centroid in the original reduced space
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            centroid_idx_in_cluster = np.argmin(distances)
            centroid_idx = indices[centroid_idx_in_cluster]
            
            # Get the 2D position of the centroid
            centroid_2d = embeddings_2d[centroid_idx]
            
            # Plot centroid with a distinct marker
            plt.scatter(centroid_2d[0], centroid_2d[1],
                       c='black', marker='*', s=300, 
                       edgecolors='yellow', linewidths=2,
                       zorder=5)
    
    # Add legend
    plt.legend(loc='best', fontsize=8)
    
    # Add labels and title
    method_name = 't-SNE' if method == 'tsne' else 'UMAP'
    plt.title(f'Query Cluster Visualization using {method_name}')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    
    # Add a note about centroid markers
    plt.text(0.02, 0.98, '★ = Cluster Centroid', 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main(spl_queries, summary_output=None, samples_output=None, 
         show_all_queries=False, min_cluster_size=MIN_CLUSTER_SIZE, num_samples=3,
         min_samples=1, cluster_selection_epsilon=0.0, cluster_selection_method='eom', 
         alpha=1.0, visualize_tsne=None, visualize_umap=None,
         tsne_perplexity=30, tsne_learning_rate=200, tsne_n_iter=1000,
         umap_n_neighbors=15, umap_min_dist=0.1, umap_metric='euclidean'):
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
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        alpha=alpha,
    )

    cluster_labels = clusterer.fit_predict(D)

    # Get clusters for visualization
    clusters, cluster_indices = get_clusters(zip(spl_queries, cluster_labels))
    
    # Generate visualizations if requested
    if visualize_tsne:
        tsne_params = {
            'perplexity': tsne_perplexity,
            'learning_rate': tsne_learning_rate,
            'n_iter': tsne_n_iter,
            'random_state': 42
        }
        visualize_clusters(reduced_embeddings, cluster_labels, clusters, cluster_indices,
                         visualize_tsne, method='tsne', **tsne_params)
    
    if visualize_umap:
        umap_params = {
            'n_neighbors': umap_n_neighbors,
            'min_dist': umap_min_dist,
            'metric': umap_metric,
            'random_state': 42
        }
        visualize_clusters(reduced_embeddings, cluster_labels, clusters, cluster_indices,
                         visualize_umap, method='umap', **umap_params)

    # Handle output files
    if show_all_queries:
        # Original behavior: single file output
        if summary_output:
            with open(summary_output, 'w', newline='') as f:
                print_clusters(zip(spl_queries, cluster_labels), f, None,
                             show_all_queries, reduced_embeddings, num_samples)
        else:
            print_clusters(zip(spl_queries, cluster_labels), sys.stdout, None,
                         show_all_queries, reduced_embeddings, num_samples)
    else:
        # Two-file output
        summary_file = open(summary_output, 'w', newline='') if summary_output else sys.stdout
        samples_file = open(samples_output, 'w', newline='') if samples_output else None
        
        try:
            print_clusters(zip(spl_queries, cluster_labels), summary_file, samples_file,
                         show_all_queries, reduced_embeddings, num_samples)
        finally:
            if summary_output and summary_file != sys.stdout:
                summary_file.close()
            if samples_output and samples_file:
                samples_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cluster queries using CodeBERT "
                                     "and HDBSCAN. Outputs two CSV files: "
                                     "1) cluster summary with centroid queries, "
                                     "2) sample queries from each cluster.")
    parser.add_argument("--input", type=str, help="Path to the CSV file containing queries.", required=True)
    parser.add_argument("--output-summary", type=str, 
                        help="Path to output CSV file for cluster summary (default: stdout)")
    parser.add_argument("--output-samples", type=str,
                        help="Path to output CSV file for sample queries (default: not generated)")
    parser.add_argument("--show-all-queries", action="store_true",
                        help="Show all queries in cluster and do not aggregate metrics (single file output)")
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE,
                        help="Minimum cluster size for HDBSCAN (default: {})".format(MIN_CLUSTER_SIZE))
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of sample queries to show per cluster (default: 3)")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="Number of samples in a neighborhood for a core point (default: 1)")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                        help="Distance threshold for merging clusters (default: 0.0)")
    parser.add_argument("--cluster-selection-method", type=str, default='eom',
                        choices=['eom', 'leaf'],
                        help="Method for selecting clusters from the tree (default: 'eom')")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Conservativeness for cluster selection (default: 1.0)")
    
    # Visualization options
    parser.add_argument("--visualize-tsne", type=str,
                        help="Path to save t-SNE visualization (e.g., tsne_plot.png)")
    parser.add_argument("--tsne-perplexity", type=float, default=30,
                        help="t-SNE perplexity parameter (default: 30)")
    parser.add_argument("--tsne-learning-rate", type=float, default=200,
                        help="t-SNE learning rate parameter (default: 200)")
    parser.add_argument("--tsne-n-iter", type=int, default=1000,
                        help="t-SNE number of iterations (default: 1000)")
    
    parser.add_argument("--visualize-umap", type=str,
                        help="Path to save UMAP visualization (e.g., umap_plot.png)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter (default: 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist parameter (default: 0.1)")
    parser.add_argument("--umap-metric", type=str, default='euclidean',
                        help="UMAP distance metric (default: 'euclidean')")
    
    args = parser.parse_args()

    spl_queries = load_queries_from_csv(args.input)
    main(spl_queries, 
         summary_output=args.output_summary,
         samples_output=args.output_samples,
         show_all_queries=args.show_all_queries, 
         min_cluster_size=args.min_cluster_size, 
         num_samples=args.num_samples,
         min_samples=args.min_samples,
         cluster_selection_epsilon=args.cluster_selection_epsilon,
         cluster_selection_method=args.cluster_selection_method,
         alpha=args.alpha,
         visualize_tsne=args.visualize_tsne,
         visualize_umap=args.visualize_umap,
         tsne_perplexity=args.tsne_perplexity,
         tsne_learning_rate=args.tsne_learning_rate,
         tsne_n_iter=args.tsne_n_iter,
         umap_n_neighbors=args.umap_n_neighbors,
         umap_min_dist=args.umap_min_dist,
         umap_metric=args.umap_metric)
