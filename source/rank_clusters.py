import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from utility import load_mean_embeddings, load_literature_barrier_embeddings

def compute_max_cosine_similarity(new_barrier_embeddings, lit_barrier_embeddings):
    """
    Computes the maximum cosine similarity between each new barrier embedding and all literature barrier embeddings.
    
    Parameters:
    - new_barrier_embeddings (dict): Dictionary mapping new barrier cluster IDs to their mean embeddings.
    - lit_barrier_embeddings (np.ndarray): Array of literature barrier embeddings.
    
    Returns:
    - dict: Dictionary mapping new barrier cluster IDs to their maximum cosine similarity scores.
    """
    max_similarity_scores = {}
    
    # Normalize all literature embeddings in advance for efficiency
    normalized_lit_embeddings = normalize(lit_barrier_embeddings)
    
    for cluster_id, embedding in new_barrier_embeddings.items():
        # Reshape embedding for sklearn (expects 2D array)
        embedding = embedding.reshape(1, -1)
        
        # Normalize the embedding
        normalized_embedding = normalize(embedding)
        
        # Compute cosine similarities with all literature embeddings
        similarities = cosine_similarity(normalized_embedding, normalized_lit_embeddings)[0]  # Get the first (and only) row
        
        # Compute the max similarity score
        max_similarity = similarities.max()
        
        # Store the max similarity score
        max_similarity_scores[cluster_id] = max_similarity
        
        print(f"Cluster ID {cluster_id}: Max Cosine Similarity = {max_similarity:.4f}")
    
    return max_similarity_scores

def aggregate_and_rank_scores(mean_similarity_scores):
    """
    Aggregates mean similarity scores into a DataFrame and ranks the new barriers.
    
    Parameters:
    - mean_similarity_scores (dict): Dictionary mapping cluster IDs to mean similarity scores.
    
    Returns:
    - pd.DataFrame: DataFrame containing cluster IDs and their mean similarity scores, sorted descendingly.
    """
    # Convert the dictionary to a DataFrame
    df_scores = pd.DataFrame(list(mean_similarity_scores.items()), columns=['Cluster_ID', 'Mean_Cosine_Similarity'])
    
    # Sort the DataFrame by 'Mean_Cosine_Similarity' in descending order
    df_scores_sorted = df_scores.sort_values(by='Mean_Cosine_Similarity', ascending=False).reset_index(drop=True)
    
    print("\nRanked New Barriers based on Mean Cosine Similarity:")
    print(df_scores_sorted)
    
    return df_scores_sorted

def save_ranked_scores(df_ranked, output_ranked_csv):
    """
    Saves the ranked similarity scores to a CSV file.
    
    Parameters:
    - df_ranked (pd.DataFrame): DataFrame containing ranked new barriers.
    - output_ranked_csv (str): Path to save the ranked scores as a CSV.
    
    Returns:
    - None
    """
    try:
        df_ranked.to_csv(output_ranked_csv, index=False)
        print(f"Ranked similarity scores saved to '{output_ranked_csv}'.")
    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")

def sort_by_descriptors(npz_path, all_data_file, output_file, file_name):
    # Define your file paths
    # file_name = "all_data_descriptors"
    # csv_file_path = f'./data/barriers_2018-2022/processed_posts_{file_name}_mean_embedding.csv'                           # Replace with your actual CSV file path
    # all_barriers_csv = f'./data/barriers_2018-2022/processed_posts_{file_name}_clusters_all.csv'                           # Replace with your actual CSV file path
    # all_clusters_file_path = f'./data/barriers_2018-2022/processed_posts_{file_name}_clusters_all.csv'                           # Replace with your actual CSV file path
    # npz_path = f'./data/embeddings/{file_name}_clustered_embeddings_mean.npz'       # Path to the filtered mean embeddings .npz file
    lit_input_file = './data/embeddings/barriers_lit_embeddings.npz'               # Path to the literature barriers .npz file
    output_ranked_csv = f'./data/{file_name}_new_barriers_ranked.csv'              # Path to save the ranked scores (optional)
    # output_new_barriers_csv = f'./data/barriers_2018-2022/{file_name}_new_barrier_clusters_with_mean.csv'              # Path to save the ranked scores (optional)
    # top_n = 10                                                          # Number of top barriers to visualize
    
    # Step Load the mean embeddings
    mean_embeddings_dict = load_mean_embeddings(npz_path)
    # Step 2: Load the filtered mean embeddings
    # filtered_mean_embeddings = load_filtered_mean_embeddings(filtered_npz_path)
   
    
    # Step 3: Load literature barrier embeddings
    lit_barrier_embeddings, lit_barrier_labels = load_literature_barrier_embeddings(lit_input_file)
    
    if lit_barrier_embeddings is None or lit_barrier_labels is None:
        print("Failed to load literature barrier embeddings. Exiting.")
        return
    
    # Optional: Normalize embeddings for better similarity measures
    # lit_barrier_embeddings = normalize(lit_barrier_embeddings, norm='l2')
    # filtered_mean_embeddings = {cluster_id: normalize(embedding.reshape(1, -1)).flatten()
    #                             for cluster_id, embedding in filtered_mean_embeddings.items()}
    
    # Step 4: Compute mean cosine similarity scores
    mean_similarity_scores = compute_max_cosine_similarity(mean_embeddings_dict, lit_barrier_embeddings)
    
    # Step 5: Aggregate and rank the new barriers
    df_ranked = aggregate_and_rank_scores(mean_similarity_scores)
    
    # Step 6: Save the ranked scores to a CSV file 
    if output_ranked_csv:
        save_ranked_scores(df_ranked, output_ranked_csv)

    all_data_df = pd.read_csv(all_data_file)
    # ranked_data_df = pd.read_csv(ranked_data_file)

    # Add the 'Rank' column based on 'Mean_Cosine_Similarity' (lower similarity, higher rank)
    df_ranked['Rank'] = df_ranked['Mean_Cosine_Similarity'].rank(method='min', ascending=True)

    # Rename 'Cluster_ID' in the second DataFrame to match the first DataFrame's column name
    df_ranked.rename(columns={'Cluster_ID': 'clusters_0.005'}, inplace=True)

    # Ensure 'clusters_0.005' is numeric (handle string representations of numbers and NaN)
    all_data_df['clusters_0.005'] = pd.to_numeric(all_data_df['clusters_0.005'], errors='coerce')
    df_ranked['clusters_0.005'] = pd.to_numeric(df_ranked['clusters_0.005'], errors='coerce')

    # Find the max value of 'clusters_0.005' in all_data_df
    max_cluster_value = all_data_df['clusters_0.005'].max()

    # Replace NaN values with max_cluster_value + 1
    all_data_df['clusters_0.005'] = all_data_df['clusters_0.005'].fillna(max_cluster_value + 1)

    # Ensure both DataFrames have 'clusters_0.005' as integers
    all_data_df['clusters_0.005'] = all_data_df['clusters_0.005'].astype(int)
    df_ranked['clusters_0.005'] = df_ranked['clusters_0.005'].astype(int)

    # Merge the two dataframes on the column 'clusters_0.005'
    merged_df = pd.merge(all_data_df, df_ranked, on='clusters_0.005', how='left')

    # Sort
    sorted_df = merged_df.sort_values(by=['Rank', 'clusters_0.005', 'clusters_0.004', 'clusters_0.003', 'clusters_0.002', 'clusters_0.001'], ascending=[True, True, True, True, True, True])

    # Save the merged DataFrame to a new CSV
    
    sorted_df.to_csv(output_file, index=False)


    print(f"Merged data saved to {output_file}")

file_name = 'all_data'
all_data_file = f'./data/OpiatesRecovery_{file_name}_secondary_clusters.csv'
input_file = './data/embeddings/cluster_embeddings_new3.npz'
output_file = './data/barriers_2018-2022/merged_data_all_new_barriers_ranked.csv'
input_file_cosine = f'./data/barriers_2018-2022/processed_posts_{file_name}_classification.csv'
output_file_cosine = './data/barriers_2018-2022/final_merged_data_new3.csv'
output_file_cosine_updated = './data/barriers_2018-2022/final_merged_data_with_filled_similarity_new3.csv'
# main_classify(file_name)
# main(file_name)
sort_by_descriptors(input_file, all_data_file, output_file)
# sort_by_descriptors(npz_path, all_data_file, output_file, file_name)