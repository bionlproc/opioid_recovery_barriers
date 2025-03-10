import ast
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from utility import read_posts, get_embedding


def create_barrier_list(input_file: str) -> list:
    """
    Reads a text file containing barriers separated by double newlines and returns a list of barriers.

    Parameters:
        input_file (str): Path to the text file containing barriers.

    Returns:
        list: A list of barrier strings.
    """
    with open(input_file, "r") as file:
        # Read the entire content of the file.
        data = file.read()
        # Split the content by double newlines to separate each barrier.
        barriers_from_file = data.strip().split("\n\n")
        print(barriers_from_file)
        return barriers_from_file


def generate_embeddings_lit(input_file: str) -> np.ndarray:
    """
    Generates embeddings for a list of barriers obtained from a literature file.

    Parameters:
        input_file (str): Path to the text file containing barriers.

    Returns:
        np.ndarray: Array of embeddings for each barrier.
    """
    barriers_from_lit = create_barrier_list(input_file)
    # Generate an embedding for each barrier using the get_embedding function.
    lit_embeddings = [get_embedding(barrier_lit) for barrier_lit in barriers_from_lit]
    return np.array(lit_embeddings)


def generate_barrier_embeddings_all(input_file: str, output_file: str) -> np.ndarray:
    """
    Reads a CSV file containing barriers, generates embeddings for each barrier,
    and saves the resulting IDs, barrier texts, and embeddings into an NPZ file.

    Parameters:
        input_file (str): Path to the CSV file with barrier data.
        output_file (str): Path where the NPZ file with embeddings will be saved.

    Returns:
        np.ndarray: Array of embeddings for all barriers.
    """
    # Load the DataFrame from the CSV file.
    data_df = read_posts(input_file)

    # Initialize lists to collect IDs, barrier texts, and embeddings.
    ids = []
    barriers = []
    embeddings = []

    # Iterate over each row in the DataFrame with a progress bar.
    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing Barriers"):
        ids.append(row['id'])                     # Collect the barrier's ID.
        barriers.append(row['barrier'])           # Collect the barrier text.
        embedding = get_embedding(row['barrier'])   # Generate the embedding.
        embeddings.append(embedding)              # Append the embedding to the list.

    # Convert lists to NumPy arrays.
    ids_array = np.array(ids)
    barriers_array = np.array(barriers)
    embeddings_array = np.array(embeddings)

    # Save the arrays to a .npz file.
    np.savez(output_file, ids=ids_array, barriers=barriers_array, embeddings=embeddings_array)
    print(f"Data saved to {output_file}")
    return embeddings_array


def generate_lit_barrier_embeddings(barriers_lit_input_file: str, barriers_lit_embedding_file: str) -> str:
    """
    Generates and saves embeddings for literature-based barriers.

    Parameters:
        barriers_lit_input_file (str): Path to the literature file containing barriers.
        barriers_lit_embedding_file (str): Path where the NPZ file for literature embeddings will be saved.

    Returns:
        str: The output file path where the literature embeddings are saved.
    """
    # Generate embeddings for the barriers listed in the literature file.
    barrier_lit_embeddings = generate_embeddings_lit(barriers_lit_input_file)
    # Read the barrier list again (for saving with embeddings).
    barriers_from_lit = create_barrier_list(barriers_lit_input_file)
    np.savez(
        barriers_lit_embedding_file,
        barriers_lit=np.array(barriers_from_lit),
        barrier_lit_embeddings=barrier_lit_embeddings
    )
    return barriers_lit_embedding_file


def keyphrase_embedding_individual(keyphrases: str) -> tuple:
    """
    Computes individual embeddings for each keyphrase provided as a string representation of a list.
    Returns both the mean embedding and a list of individual embeddings.

    Parameters:
        keyphrases (str): A string representing a list of keyphrases (e.g., "[key1, key2, ...]").

    Returns:
        tuple: A tuple containing:
            - mean_embedding (np.ndarray): The mean embedding of all keyphrases.
            - embeddings (list): List of embeddings for each keyphrase.
    """
    embeddings = []
    # Convert the string representation of keyphrases into a Python list.
    keyphrases = ast.literal_eval(keyphrases)
    print(f"Keyphrases: {keyphrases}, Type: {type(keyphrases)}")
    
    # Compute the embedding for each keyphrase.
    for keyphrase in keyphrases:
        print(f'Keyphrase: {keyphrase}')
        response = get_embedding(keyphrase)
        embeddings.append(np.array(response))
    
    # Compute the mean embedding across all keyphrases.
    mean_embedding = np.mean(embeddings, axis=0)
    time.sleep(1)
    return mean_embedding, embeddings


def save_keyphrase_embeddings_individual(file_name: str) -> str:
    """
    Reads keyphrase data from a CSV file, computes individual and mean embeddings for each cluster,
    and saves the results to NPZ files.

    Parameters:
        file_name (str): Base name used to construct the input file path and output file names.

    Returns:
        str: The path to the NPZ file containing the mean keyphrase embeddings.
    """
    # Define input and output file paths.
    input_file = f'./data/OpiatesRecovery_{file_name}_new_barriers_keyphrases.csv'
    output_npz = f'./data/embeddings/{file_name}_keyphrase_embeddings_individual.npz'
    output_mean_npz = f'./data/embeddings/{file_name}_keyphrase_embeddings_individual_mean.npz'

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(input_file)

    # Initialize lists and dictionary to store embeddings.
    embedding_dict = {}
    cluster_id_list = []
    mean_embedding_list = []

    # Iterate over each row with a progress bar.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        cluster_id = row['cluster_id']
        keyphrases = row['keyphrases']
        # Compute mean and individual embeddings for the keyphrases.
        mean_embedding, embeddings = keyphrase_embedding_individual(keyphrases)
        cluster_id_list.append(cluster_id)
        mean_embedding_list.append(mean_embedding)
        
        # Convert the list of embeddings into a 2D NumPy array.
        if embeddings:
            embedding_array = np.vstack(embeddings)
        else:
            # Use an empty array if there are no embeddings (adjust embedding_dim as needed).
            embedding_array = np.empty((0, 768))
        embedding_dict[str(cluster_id)] = embedding_array

    # Save individual embeddings and mean embeddings to separate NPZ files.
    np.savez(output_npz, **embedding_dict)
    np.savez(output_mean_npz, clusters=np.array(cluster_id_list), mean_keyphrase_embeddings=np.array(mean_embedding_list))
    return output_mean_npz


def generate_cluster_descriptor_embeddings(input_file: str, output_file: str) -> None:
    """
    Reads a CSV file with cluster descriptors, computes an embedding for each descriptor,
    and saves the mapping of clusters to their embeddings in an NPZ file.

    Parameters:
        input_file (str): Path to the CSV file containing 'Cluster' and 'Descriptor' columns.
        output_file (str): Path where the NPZ file with cluster embeddings will be saved.
    """
    data = pd.read_csv(input_file)
    cluster_embedding_dict = {}

    # Process each row to compute and store the descriptor embedding.
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        cluster = str(row['Cluster'])
        descriptor = row['Descriptor']
        print(descriptor)
        embedding = np.array(get_embedding(descriptor))
        cluster_embedding_dict[cluster] = embedding

    # Save the cluster-to-embedding dictionary to an NPZ file.
    np.savez(output_file, **cluster_embedding_dict)
    print(f"Cluster-embedding dictionary saved to {output_file}")


# =============================================================================
# Example Execution Block 
# =============================================================================
# Uncomment and adjust file paths to run the functions as needed.

# file_name = 'all_data'
# input_file = './data/OpiatesRecovery_code_test_posts_barriers_list.csv'
# output_file = './data/embeddings/barrier_embeddings.npz'
# barriers_lit_input_file = './data/barriers_to_recovery_lit.txt'
# barriers_lit_embedding_file = './data/embeddings/barriers_lit_embeddings.npz'

# generate_lit_barrier_embeddings(barriers_lit_input_file, barriers_lit_embedding_file)
# generate_barrier_embeddings_all(input_file, output_file)
# save_keyphrase_embeddings_individual(file_name)
# generate_cluster_descriptor_embeddings('./data/cluster_descriptors.csv', './data/embeddings/cluster_descriptor_embeddings.npz')
