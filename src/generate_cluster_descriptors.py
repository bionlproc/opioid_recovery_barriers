import time
import numpy as np
import pandas as pd
from openai import ChatCompletion, OpenAI
from tqdm import tqdm
from utility import get_gpt4_response, get_embedding

# Initialize tqdm pandas extension for progress_apply functionality
tqdm.pandas()

def generate_cluster_descriptors(csv_file, output_file, count_threshold=10, word_limit=50):
    """
    Generate cluster descriptors for opioid use disorder recovery barriers.

    This function reads a CSV file containing clusters of barriers, filters the clusters based on a 
    minimum count threshold, groups the barriers by cluster, and then uses GPT-4 to generate a concise 
    descriptor for each cluster. The descriptors capture the primary themes and challenges associated with the barriers.
    
    Parameters:
        csv_file (str): Path to the input CSV file containing clusters and barriers.
        output_file (str): Path to the output CSV file where generated descriptors will be saved.
        count_threshold (int, optional): Minimum number of occurrences required for a cluster to be processed. Defaults to 10.
        word_limit (int, optional): Maximum word limit for the descriptor (not currently enforced in the prompt). Defaults to 50.
    
    Returns:
        None
    """
    # Step 1: Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file)

    # Step 2: Filter the DataFrame to include only clusters with a count greater than or equal to the threshold.
    filtered_df = df[df['count'] >= count_threshold]

    # Step 3: Group the filtered DataFrame by the 'clusters_0.005' column and aggregate barriers into lists.
    clusters = filtered_df.groupby('clusters_0.005')['barriers'].apply(list)

    # Step 4: Generate descriptors for each cluster using GPT-4.
    cluster_descriptors = []
    for cluster, barriers in clusters.items():
        # Construct a prompt with the list of barriers.
        prompt = (
            "Given the following list of barriers to opioid use disorder recovery, write a single, concise descriptor "
            "in two to three sentences that encapsulates the primary themes and challenges. Present the descriptor as "
            "one unified paragraph without dividing it into separate points. "
            f"The barriers are:\n{', '.join(barriers)}"
        )

        # Create the message payload for GPT-4.
        messages_q1 = [
            {"role": "system", "content": "You are an expert in opioid use disorder recovery."},
            {"role": "user", "content": f"{prompt}.\n"}
        ]

        # Get the descriptor response from GPT-4.
        descriptor = get_gpt4_response(messages_q1)
        print(f'Descriptor for cluster {cluster}: {descriptor}')
        
        # Append the cluster ID and its descriptor to the list.
        cluster_descriptors.append({'Cluster': cluster, 'Descriptor': descriptor})
        time.sleep(2)  # Pause briefly to respect API rate limits.

    # Step 5: Save the list of descriptors to a CSV file.
    output_df = pd.DataFrame(cluster_descriptors)
    output_df.to_csv(output_file, index=False)
    print(f"Descriptors saved to {output_file}")

# Example Usage:
# Uncomment and adjust file paths as needed.
# file_name = 'all_data'
# cluster_output_file = f'./data/OpiatesRecovery_{file_name}_secondary_clusters.csv'
# cluster_descriptor_file = f'./data/OpiatesRecovery_{file_name}_cluster_descriptors.csv'
#
# generate_cluster_descriptors(cluster_output_file, cluster_descriptor_file)

