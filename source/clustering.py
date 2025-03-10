import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
import plotly.figure_factory as ff


def get_low_conf_points(embeddings, clusters, low_conf_threshold):
    """
    Compute silhouette scores for each embedding and identify low-confidence points.

    Parameters:
        embeddings (np.ndarray): Array of embedding vectors.
        clusters (np.ndarray): Cluster labels for each embedding.
        low_conf_threshold (float): Threshold below which points are considered low-confidence.

    Returns:
        tuple:
            - silhouette_vals (np.ndarray): Silhouette score for each embedding.
            - low_conf_points (np.ndarray): Boolean array indicating low-confidence points.
    """
    # Calculate the silhouette scores for each point in the embedding space.
    silhouette_vals = silhouette_samples(embeddings, clusters)
    # Identify points with silhouette score below the threshold.
    low_conf_points = silhouette_vals < low_conf_threshold
    return silhouette_vals, low_conf_points


def compute_mean_cluster_embeddings(npz_file_path, save_path=None):
    """
    Computes the mean embedding for each cluster from the provided .npz file.

    The .npz file should contain 'cluster_labels' and 'embeddings'.

    Parameters:
        npz_file_path (str): Path to the .npz file containing cluster labels and embeddings.
        save_path (str or None): If provided, saves the mean embeddings dictionary to this path.

    Returns:
        dict: Dictionary where keys are cluster labels (as strings) and values are mean embedding vectors.
    """
    # Load clustering data from the .npz file.
    data = np.load(npz_file_path)
    cluster_labels = data['cluster_labels']
    embeddings = data['embeddings']

    # Validate that the number of labels matches the number of embeddings.
    if cluster_labels.shape[0] != embeddings.shape[0]:
        print("Error: The number of cluster labels does not match the number of embeddings.")
        return

    # Find the unique cluster identifiers.
    unique_clusters = np.unique(cluster_labels)
    print(f"Found {len(unique_clusters)} unique clusters.")

    # Compute and store the mean embedding for each cluster.
    mean_embeddings_dict = {}
    for cluster in unique_clusters:
        cluster_embeddings = embeddings[cluster_labels == cluster]
        mean_embedding = cluster_embeddings.mean(axis=0)
        mean_embeddings_dict[str(cluster)] = mean_embedding

    # Optionally save the mean embeddings dictionary to a file.
    if save_path:
        np.savez(save_path, **mean_embeddings_dict)
        print(f"Mean embeddings saved to {save_path}")

    return mean_embeddings_dict


def save_barrier_embeddings(ids, barriers, embeddings, labels, output_file):
    """
    Save barrier information and embeddings to an NPZ file.

    Parameters:
        ids (list or np.ndarray): List of barrier IDs.
        barriers (list or np.ndarray): List of barrier text strings.
        embeddings (np.ndarray): Array of barrier embeddings.
        labels (list or np.ndarray): Cluster labels for each barrier.
        output_file (str): File path to save the NPZ file.
    """
    np.savez(
        output_file,
        ids=np.array(ids),
        barriers=np.array(barriers),
        embeddings=np.array(embeddings),
        cluster_labels=np.array(labels)
    )


def cluster_embeddings_with_agglo_cosine(embeddings, distance_threshold=5.0, n_clusters=None):
    """
    Perform agglomerative clustering on embeddings using cosine distance.

    Parameters:
        embeddings (np.ndarray): Array of embedding vectors.
        distance_threshold (float): Threshold to decide cluster formation.
        n_clusters (int or None): Number of clusters to form; if None, use distance_threshold.

    Returns:
        tuple:
            - clusters (np.ndarray): Cluster labels assigned to each embedding.
            - labels (np.ndarray): Additional cluster label attribute from the clustering instance.
    """
    # Create an Agglomerative Clustering instance with cosine metric.
    agg_cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    # Fit the model and predict cluster labels.
    clusters = agg_cluster.fit_predict(embeddings)
    labels = agg_cluster.labels_
    return clusters, labels


def initial_clustering(new_embeddings_file, file_name, distance_threshold=0.50):
    """
    Perform initial clustering on new barrier embeddings and save results.

    Parameters:
        new_embeddings_file (str): Path to the .npz file containing new barrier embeddings.
        file_name (str): Base name for output files.
        distance_threshold (float): Distance threshold for clustering.

    Returns:
        tuple:
            - cluster_output_file (str): CSV file path with clustering results.
            - mean_embedding_output_file (str): NPZ file path with mean cluster embeddings.
    """
    print("\nStarting initial clustering")
    cluster_output_file = f'./data/OpiatesRecovery_{file_name}_initial_clusters.csv'
    embedding_output_file = f"./data/embeddings/{file_name}_initial_cluster_embeddings.npz"
    mean_embedding_output_file = f"./data/embeddings/{file_name}_initial_cluster_embeddings_mean.npz"
    low_conf_threshold = 0.2

    # Load new barrier embeddings and related data.
    data_new = np.load(new_embeddings_file)
    ids = data_new['id']
    embeddings = data_new['new_barriers_embeddings']
    barriers = data_new['new_barriers']

    # Normalize embeddings for clustering.
    X_normalized = normalize(embeddings, norm='l2')

    print("\nClustering with agglomerative clustering")
    clusters, labels = cluster_embeddings_with_agglo_cosine(X_normalized, distance_threshold)
    save_barrier_embeddings(ids, barriers, embeddings, labels, embedding_output_file)
    n_clusters = len(np.unique(labels))
    print(f"\nNumber of clusters: {n_clusters}")

    # Compute silhouette scores and identify low-confidence points.
    silhouette_vals, low_conf_points = get_low_conf_points(embeddings, clusters, low_conf_threshold)
    silhouette_score1 = silhouette_score(X_normalized, labels, metric='cosine')
    print(f'\nSilhouette Score for combined embeddings: {silhouette_score1}')

    # Save clustering results to a CSV file.
    df = pd.DataFrame({
        'id': ids,
        'barriers': barriers,
        'assigned_cluster': clusters,
        'silhouette_score': silhouette_vals
    })
    df.to_csv(cluster_output_file, index=False)
    print(f"Clustering is completed. Output saved at {cluster_output_file}")

    print("\nComputing mean embeddings")
    compute_mean_cluster_embeddings(embedding_output_file, mean_embedding_output_file)

    return cluster_output_file, mean_embedding_output_file


# ========================
# Secondary Clustering
# ========================

def filter_mean_embeddings(input_file, cluster_input=None):
    """
    Filter and extract mean embeddings from clusters.

    Parameters:
        input_file (str): Path to the NPZ file containing mean embeddings (keys are cluster labels).
        cluster_input (str or None): Optional CSV file with filtered cluster IDs.

    Returns:
        tuple:
            - mean_embeddings (np.ndarray): 2D array of mean embeddings for selected clusters.
            - clusters_filtered (np.ndarray): Array of filtered cluster IDs.
    """
    data = np.load(input_file)
    # Get all cluster keys from the NPZ file.
    cluster_keys = data.files
    # Convert keys (saved as strings) to integers.
    clusters = np.array([int(key) for key in cluster_keys])
    print(f"Number of original clusters: {len(clusters)}")
    print("\n", clusters)

    if cluster_input is not None:
        print(cluster_input)
        df = pd.read_csv(cluster_input)
        clusters_filtered = df['cluster_id'].values
    else:
        clusters_filtered = clusters

    print(f"\nFiltered clusters: {clusters_filtered}. Number of clusters: {len(clusters_filtered)}")
    # Stack mean embeddings for the filtered clusters into a 2D array.
    mean_embeddings = np.stack([data[str(cluster)] for cluster in clusters_filtered])
    return mean_embeddings, clusters_filtered


def load_weighted_cluster_keyphrase_embeddings(mean_embedding_file, filtered_cluster_file, npz_file_path, barrier_weight=0.5):
    """
    Combine barrier and keyphrase embeddings using a weighted sum.

    Parameters:
        mean_embedding_file (str): Path to NPZ file containing mean barrier embeddings.
        filtered_cluster_file (str): CSV file used to filter cluster IDs.
        npz_file_path (str): Path to NPZ file with keyphrase embeddings and cluster labels.
        barrier_weight (float): Weight assigned to barrier embeddings; keyphrase weight is 1 - barrier_weight.

    Returns:
        dict: Dictionary mapping cluster labels to a list of weighted embeddings.
    """
    keyphrase_weight = 1 - barrier_weight
    barrier_embeddings, ori_labels = filter_mean_embeddings(mean_embedding_file, filtered_cluster_file)

    # Load keyphrase embeddings and associated cluster labels.
    data = np.load(npz_file_path)
    keyphrase_embeddings = data['mean_keyphrase_embeddings']  # Shape: (num_samples, embedding_dim)
    cluster_labels = data['clusters']  # Shape: (num_samples,)

    # Compute weighted embeddings for each cluster.
    weighted_embeddings = [
        barrier_weight * barrier + keyphrase_weight * keyphrase
        for barrier, keyphrase in zip(barrier_embeddings, keyphrase_embeddings)
    ]

    # Group the weighted embeddings by their cluster labels.
    cluster_dict = defaultdict(list)
    print(f'Length of cluster labels: {len(cluster_labels)} and weighted embeddings: {len(weighted_embeddings)}')
    for label, embedding in zip(cluster_labels, weighted_embeddings):
        cluster_dict[label].append(embedding)

    return dict(cluster_dict)


def compute_representative_embeddings(cluster_dict):
    """
    Compute the representative (mean) embedding for each cluster from the provided cluster dictionary.

    Parameters:
        cluster_dict (dict): Dictionary mapping cluster IDs to lists of embeddings.

    Returns:
        dict: Dictionary mapping each cluster ID to its mean embedding vector.
    """
    representative_embeddings = {}
    for cluster_id, embeddings in cluster_dict.items():
        embeddings_array = np.array(embeddings)
        mean_embedding = embeddings_array.mean(axis=0)
        representative_embeddings[cluster_id] = mean_embedding
    return representative_embeddings


def interactive_dendrogram(distance_matrix, linked, weight=0.5):
    """
    Create and display an interactive dendrogram using the distance matrix and linkage results.

    Parameters:
        distance_matrix (np.ndarray): Pairwise distance matrix.
        linked (np.ndarray): Linkage matrix obtained from hierarchical clustering.
        weight (float): Weight used in the clustering (used for output file naming).
    """
    # Create an interactive dendrogram using Plotly.
    fig = ff.create_dendrogram(distance_matrix, linkagefun=lambda x: linked, color_threshold=0.7)
    fig.update_layout(width=12000, height=1000)  # Adjust the figure size as needed
    fig.update_traces(
        textfont=dict(size=1),  # Adjust font size for labels
        selector=dict(type='scatter', mode='text')
    )

    # Save the dendrogram as an HTML file.
    output_path_html = f'./data/images/interactive_dendrogram_keyphrase_{weight}.html'
    fig.write_html(output_path_html)
    print(f"Interactive dendrogram saved successfully at {output_path_html}")

    # Display the figure (useful in Jupyter Notebook environments).
    fig.show()


def secondary_clustering_with_keyphrases(mean_embedding_file, filtered_cluster_file, keyphrase_npz_path, weight, distance_threshold, cluster_input_file, cluster_output_file):
    """
    Perform secondary clustering by combining barrier and keyphrase embeddings.

    Parameters:
        mean_embedding_file (str): NPZ file with initial mean barrier embeddings.
        filtered_cluster_file (str): CSV file with filtered cluster IDs.
        keyphrase_npz_path (str): NPZ file with keyphrase embeddings.
        weight (float): Weight for barrier embeddings (keyphrase weight = 1 - weight).
        distance_threshold (float): Threshold for forming flat clusters.
        cluster_input_file (str): CSV file with initial clustering results.
        cluster_output_file (str): Output CSV file path for secondary clustering results.
    """
    # Load and compute weighted embeddings by combining barrier and keyphrase data.
    new_barriers_embeddings = load_weighted_cluster_keyphrase_embeddings(
        mean_embedding_file, filtered_cluster_file, keyphrase_npz_path, barrier_weight=weight
    )
    # Compute representative (mean) embeddings for each cluster.
    representative_embeddings = compute_representative_embeddings(new_barriers_embeddings)
    cluster_ids = list(representative_embeddings.keys())
    embedding_matrix = np.vstack([representative_embeddings[cluster_id] for cluster_id in cluster_ids])

    # Compute the cosine similarity matrix and convert it to a distance matrix.
    similarity_matrix = cosine_sim(embedding_matrix)
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering using the distance matrix.
    linked = linkage(distance_matrix, method='average', metric='cosine')

    # Form flat clusters based on the specified distance threshold.
    cluster_labels = fcluster(linked, t=distance_threshold, criterion='distance')
    # Generate additional clustering labels at various thresholds (for analysis purposes)
    cluster_labels1 = fcluster(linked, t=0.001, criterion='distance')
    cluster_labels2 = fcluster(linked, t=0.002, criterion='distance')
    cluster_labels3 = fcluster(linked, t=0.003, criterion='distance')
    cluster_labels4 = fcluster(linked, t=0.004, criterion='distance')
    cluster_labels5 = fcluster(linked, t=0.005, criterion='distance')

    # Create a DataFrame to compare clustering at different thresholds.
    df = pd.DataFrame({
        'original_clusters': cluster_ids,
        'clusters_0.001': cluster_labels1,
        'clusters_0.002': cluster_labels2,
        'clusters_0.003': cluster_labels3,
        'clusters_0.004': cluster_labels4,
        'clusters_0.005': cluster_labels5,
    })

    # Merge with the initial clustering results.
    cluster_df = pd.read_csv(cluster_input_file)
    merged_df = pd.merge(cluster_df, df, left_on='assigned_cluster', right_on='original_clusters', how='left')
    merged_df['count'] = merged_df.groupby('clusters_0.005')['clusters_0.005'].transform('count')

    # Sort the merged DataFrame to prioritize clusters.
    sorted_df = merged_df.sort_values(
        by=['count', 'clusters_0.005', 'clusters_0.004', 'clusters_0.003', 'clusters_0.002', 'clusters_0.001'],
        ascending=[True, True, True, True, True, True]
    )
    sorted_df = sorted_df[sorted_df['clusters_0.005'].notna()]
    sorted_df.to_csv(cluster_output_file, index=False)

    # Calculate and print the silhouette score.
    score = silhouette_score(distance_matrix, cluster_labels, metric='cosine')
    print(f"Silhouette Score at threshold {distance_threshold} and weight {weight}: {score:.4f}")
    num_clusters = len(np.unique(cluster_labels))
    print(f"Number of clusters formed at threshold {distance_threshold} with weight {weight}: {num_clusters}")

    # Display an interactive dendrogram of the clusters.
    interactive_dendrogram(distance_matrix, linked, weight=weight)


# =============================================================================
# Main Execution Block
# =============================================================================
# if __name__ == "__main__":
#     # Initial clustering of new barrier embeddings.
#     file_name = "all_data"
#     input_file = f'./data/embeddings/{file_name}_new_barrier_embeddings.npz'
#     distance_threshold_initial = 0.43
#     initial_clustering(input_file, file_name, distance_threshold_initial)

#     # Parameters for secondary clustering using keyphrase embeddings.
#     keyphrase_npz_path = f'./data/embeddings/{file_name}_keyphrase_embeddings_individual_mean.npz'
#     mean_embedding_file = f"./data/embeddings/{file_name}_initial_cluster_embeddings_mean.npz"
#     filtered_cluster_file = f'data/OpiatesRecovery_{file_name}_new_barriers_keyphrases.csv'
#     cluster_input_file = f'./data/OpiatesRecovery_{file_name}_initial_clusters.csv'
#     cluster_output_file = f'./data/OpiatesRecovery_{file_name}_secondary_clusters.csv'
#     weight = 0.3
#     distance_threshold_secondary = 0.005

#     secondary_clustering_with_keyphrases(
#         mean_embedding_file,
#         filtered_cluster_file,
#         keyphrase_npz_path,
#         weight,
#         distance_threshold_secondary,
#         cluster_input_file,
#         cluster_output_file
#     )
