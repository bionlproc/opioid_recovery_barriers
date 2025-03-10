import numpy as np
import pandas as pd
import tqdm

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def classify_barriers(barrier_embeddings, barrier_lit_embeddings, file_name, ids, barriers, barrier_lit_labels, threshold=0.5):
    """
    Compare barrier embeddings with literature barrier embeddings and classify each barrier.

    The function works in two modes:
      1. 'classification': Maps each barrier with a literature barrier based on maximum cosine similarity.
      2. 'mean_embedding': (Conceptual) Maps a mean embedding (e.g., a cluster embedding) with a literature barrier. 

    Parameters:
        barrier_embeddings (np.ndarray): Array of embeddings for barriers.
        barrier_lit_embeddings (np.ndarray): Array of literature barrier embeddings.
        file_name (str): Base file name used for output files.
        ids (np.ndarray): Array of barrier IDs.
        barriers (np.ndarray): Array of barrier texts.
        barrier_lit_labels (list or np.ndarray): Labels for the literature barriers.
        threshold (float, optional): Minimum cosine similarity threshold to assign a literature label. Defaults to 0.5.

    Returns:
        tuple: A tuple containing:
            - classified_labels (list): List of labels assigned to each barrier.
            - output_npz (str): File path of the saved NPZ file with new barrier embeddings.
    """
    # Define output file paths.
    output_csv = f'./data/OpiatesRecovery_{file_name}.csv'
    output_new_csv = f'./data/OpiatesRecovery_{file_name}_new_barriers.csv'
    output_npz = f'./data/embeddings/{file_name}_new_barrier_embeddings.npz'

    classified_labels = []      # To store final classification labels for each barrier.
    rows = []                   # To store detailed results for each barrier.
    new_barriers_rows = []      # To store details of barriers classified as "new_barrier".
    new_barriers_ids = []       # To store IDs of barriers that are considered new.
    new_barriers = []           # To store text of new barriers.
    new_barriers_embeddings = []  # To store embeddings of new barriers.
   
    # Loop through each barrier embedding.
    for idx, barrier_embedding in enumerate(barrier_embeddings):
        max_similarity = -1
        best_label = None
        label_id = 999  # Default literature label ID if no match is found.
        
        # Compare the current barrier embedding against all literature barrier embeddings.
        for i, lit_embedding in enumerate(barrier_lit_embeddings):
            similarity = cosine_similarity(barrier_embedding, lit_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_label = barrier_lit_labels[i]
                label_id = i
        
        # If the maximum similarity exceeds the threshold, use the literature label.
        if max_similarity >= threshold:
            classified_labels.append(best_label)
        else:
            # Otherwise, mark as a new barrier.
            classified_labels.append("new_barrier")
            # Store details for the new barrier.
            new_barriers_ids.append(ids[idx])
            new_barriers.append(barriers[idx])
            new_barriers_embeddings.append(barrier_embedding)
            new_barriers_rows.append({
                'id': ids[idx],
                'barrier': barriers[idx],           # Original barrier text.
                'classified_lit_barrier': best_label, # Closest matching literature barrier.
                'classified': "new_barrier"           # Indicate new barrier classification.
            })
            best_label = "new_barrier"
        
        # Record the classification details for the current barrier.
        rows.append({
            'id': ids[idx],
            'barrier': barriers[idx],            # Original barrier text.
            'classified_lit_barrier': best_label,  # Assigned label.
            'lit_barrier_id': label_id,
            'cosine_similarity': max_similarity  # Similarity score.
        })
            
    # Create a DataFrame from the classification results and save to CSV.
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(output_csv, index=False)
    
    # Save new barrier details to an NPZ file.
    np.savez(output_npz,
             id=np.array(new_barriers_ids),
             new_barriers=np.array(new_barriers),
             new_barriers_embeddings=new_barriers_embeddings)
    
    # Create a DataFrame for new barriers and save to a separate CSV file.
    new_barriers_df = pd.DataFrame(new_barriers_rows)
    print(new_barriers_df)
    new_barriers_df.to_csv(output_new_csv, index=False)
    
    print(f'New barriers saved at {output_npz}')
    return classified_labels, output_npz


def classify_with_lit(lit_input_file, barriers_input_file, file_name, classification_threshold=0.5):
    """
    Load literature barrier embeddings and barrier embeddings, then classify the barriers.

    Parameters:
        lit_input_file (str): Path to the NPZ file containing literature barrier embeddings and labels.
        barriers_input_file (str): Path to the NPZ file containing barrier IDs, texts, and embeddings.
        file_name (str): Base file name used for output files.
        classification_threshold (float, optional): Threshold for cosine similarity. Defaults to 0.5.

    Returns:
        str: File path to the NPZ file with new barrier embeddings.
    """
    # Load literature barrier data.
    lit_data = np.load(lit_input_file)
    barrier_lit_embeddings = lit_data['barrier_lit_embeddings']
    barrier_lit_labels = lit_data['barriers_lit']  # Literature barrier labels.
    
    # Load barrier data.
    data = np.load(barriers_input_file)
    ids = data['ids']
    barrier_embeddings = data['embeddings']
    barriers = data['barriers']
    
    # Classify barriers against literature embeddings.
    _, new_embeddings_file = classify_barriers(
        barrier_embeddings,
        barrier_lit_embeddings,
        file_name,
        ids,
        barriers,
        barrier_lit_labels,
        classification_threshold
    )
    
    return new_embeddings_file


# =============================================================================
# Example Execution Block 
# =============================================================================
# Uncomment and adjust file paths to run the classification functions as needed.
#
# lit_input_file = './data/embeddings/barriers_lit_embeddings.npz'
# barriers_input_file = './data/embeddings/barrier_embeddings.npz'
# classification_threshold = 0.55
#
# # Perform classification and obtain the file with new barrier embeddings.
# new_embeddings_file = classify_with_lit(lit_input_file, barriers_input_file, 'test_posts', classification_threshold)
# print("New embeddings file:", new_embeddings_file)
