from extract_barriers import extract_finalized_barriers
from generate_embeddings import generate_lit_barrier_embeddings, generate_barrier_embeddings_all, save_keyphrase_embeddings_individual
from classification import classify_with_lit
from barriers_data_cleaning import generate_barrier_df
from clustering import initial_clustering, secondary_clustering_with_keyphrases
from generate_keyphrases import classify_barrier_no_barrier
from generate_cluster_descriptors import generate_cluster_descriptors

if __name__ == "__main__":
    # =============================================================================
    # File Paths and Parameters
    # =============================================================================
    file_name = "all_data"
    
    # Input files for posts and literature barriers.
    input_posts = './data/OpiatesRecovery_code_test_posts.csv'
    barriers_lit_input_file = './data/barriers_to_recovery_lit.txt'
    
    # Intermediate and output files for barriers.
    finalized_barriers_file = './data/OpiatesRecovery_code_test_posts_finalized_barriers.csv'
    individual_barriers_file = './data/OpiatesRecovery_code_test_posts_barriers_list.csv'
    barrier_embedding_file = './data/embeddings/barrier_embeddings.npz'
    
    # Output files for literature barrier embeddings.
    barriers_lit_embedding_file = './data/embeddings/barriers_lit_embeddings.npz'
    
    # Output file for new barrier embeddings after classification.
    new_barrier_embedding_file = f'./data/embeddings/{file_name}_new_barrier_embeddings.npz'
    
    # Clustering and descriptor files.
    initial_cluster_file = None  # Will be set after initial clustering
    cluster_output_file = f'./data/OpiatesRecovery_{file_name}_secondary_clusters.csv'
    cluster_descriptor_file = f'./data/OpiatesRecovery_{file_name}_cluster_descriptors.csv'
    
    # Thresholds and weights for classification and clustering.
    classification_threshold = 0.55
    initial_distance_threshold = 0.43
    weight = 0.3
    secondary_distance_threshold = 0.005

    # # =============================================================================
    # # Step 1: Extract Finalized Barriers
    # # =============================================================================
    # # Reads posts and extracts finalized barriers, saving the output to a CSV file.
    # print("Step 1: Extracting and finalizing barriers...")
    # extract_finalized_barriers(input_posts, finalized_barriers_file)

    # # =============================================================================
    # # Step 2: Generate Individual Barriers DataFrame
    # # =============================================================================
    # # Cleans and splits the finalized barriers into individual barrier entries.
    # print("Step 2: Generating individual barriers DataFrame...")
    # generate_barrier_df(finalized_barriers_file, individual_barriers_file)

    # # =============================================================================
    # # Step 3: Generate Literature Barrier Embeddings
    # # =============================================================================
    # # Create embeddings from a literature text file of barriers.
    # print("Step 3: Generating literature barrier embeddings...")
    # barriers_lit_embedding_file = generate_lit_barrier_embeddings(
    #     barriers_lit_input_file, barriers_lit_embedding_file
    # )

    # # =============================================================================
    # # Step 4: Generate Barrier Embeddings from Posts
    # # =============================================================================
    # # Generate embeddings for individual barriers and save them to an NPZ file.
    # print("Step 4: Generating barrier embeddings from posts...")
    # generate_barrier_embeddings_all(individual_barriers_file, barrier_embedding_file)

    # # =============================================================================
    # # Step 5: Classify Barriers Using Literature Embeddings
    # # =============================================================================
    # # Compare barrier embeddings with literature embeddings to classify them.
    # print("Step 5: Classifying barriers using literature embeddings...")
    # new_barrier_embedding_file = classify_with_lit(
    #     barriers_lit_embedding_file, barrier_embedding_file, 'test_posts', classification_threshold
    # )

    # # =============================================================================
    # # Step 6: Initial Clustering of New Barrier Embeddings
    # # =============================================================================
    # # Perform initial clustering on the new barrier embeddings.
    # print("Step 6: Performing initial clustering of new barrier embeddings...")
    # initial_cluster_file, mean_embedding_file = initial_clustering(
    #     new_barrier_embedding_file, file_name, initial_distance_threshold
    # )

    # # =============================================================================
    # # Step 7: Filter Clusters with Keyphrase Classification
    # # =============================================================================
    # # Identify clusters that actually contain meaningful barriers using keyphrase classification.
    # print("Step 7: Filtering clusters with keyphrase classification...")
    # filtered_cluster_file = classify_barrier_no_barrier(file_name)

    # # =============================================================================
    # # Step 8: Generate Keyphrase Embeddings for Clusters
    # # =============================================================================
    # # Compute individual keyphrase embeddings for each cluster.
    # print("Step 8: Generating keyphrase embeddings for clusters...")
    # keyphrase_npz_path = save_keyphrase_embeddings_individual(file_name)

    # =============================================================================
    # Step 9: Secondary Clustering with Combined Keyphrase Embeddings
    # =============================================================================
    # Refine clusters by combining barrier and keyphrase embeddings.

    # keyphrase_npz_path = f'./data/embeddings/{file_name}_keyphrase_embeddings_individual_mean.npz'
    # mean_embedding_file = f"./data/embeddings/{file_name}_initial_cluster_embeddings_mean.npz"
    # filtered_cluster_file = f'data/OpiatesRecovery_{file_name}_new_barriers_keyphrases.csv'
    # initial_cluster_file = f'./data/OpiatesRecovery_{file_name}_initial_clusters.csv'
    # cluster_output_file = f'./data/OpiatesRecovery_{file_name}_secondary_clusters.csv'

    # print("Step 9: Performing secondary clustering with weighted keyphrase embeddings...")
    # secondary_clustering_with_keyphrases(
    #     mean_embedding_file,
    #     filtered_cluster_file,
    #     keyphrase_npz_path,
    #     weight,
    #     secondary_distance_threshold,
    #     initial_cluster_file,
    #     cluster_output_file
    # )

    # =============================================================================
    # Step 10: Generate Cluster Descriptors
    # =============================================================================
    # Generate a human-readable descriptor for each cluster based on the secondary clustering output.
    print("Step 10: Generating cluster descriptors...")
    temp_file = './data/temp_clusters.csv'
    generate_cluster_descriptors(temp_file, cluster_descriptor_file, count_threshold=5)
    # generate_cluster_descriptors(cluster_output_file, cluster_descriptor_file, count_threshold=10)

    print("Pipeline execution completed successfully.")
