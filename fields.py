from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
import torch

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
    "search index=web sourcetype=apache_access status=500",
    "stats count by user",
    "stats sum(bytes) by host",
    "index=main sourcetype=syslog ERROR",
    "index=web sourcetype=apache_error ERROR"
]

embeddings = [get_embedding(query) for query in spl_queries]
embeddings = np.concatenate(embeddings) # Combine into a single array

# Dimensionality Reduction (PCA)
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)

# HDBSCAN Clustering
clusterer = HDBSCAN(min_cluster_size=2, metric='cosine')
cluster_labels = clusterer.fit_predict(reduced_embeddings)

# Print results
for i, label in enumerate(cluster_labels):
    print(f"Query: {spl_queries[i//1]} Cluster: {label}") #Corrected indexing
