import ast
import sys
import pandas as pd
from utility import get_gpt4_response

def generate_clustered_keyphrase_prompt(df):
    """
    Generate a prompt by grouping barriers by their assigned clusters.

    This function groups the DataFrame by 'assigned_cluster' and concatenates the list of barriers for each cluster
    into a formatted prompt that can be provided to GPT-4.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'assigned_cluster' and 'barriers' columns.

    Returns:
        str: A formatted prompt string with cluster IDs and their corresponding barriers.
    """
    prompt = ''
    # Group the DataFrame by 'assigned_cluster' and aggregate barriers into a list.
    grouped = df.groupby('assigned_cluster')['barriers'].apply(list).reset_index()
    for idx, row in grouped.iterrows():
        # Append cluster details to the prompt.
        prompt += f"cluster_id: {row['assigned_cluster']}, barriers: {row['barriers']}\n"
    return prompt

def clean_json_string(json_string):
    """
    Clean the JSON string returned by GPT-4 by removing formatting markers.

    This function removes unwanted leading/trailing markers such as "```json" or "```" from the GPT-4 response.

    Parameters:
        json_string (str): The raw JSON string from GPT-4.

    Returns:
        str: The cleaned JSON string.
    """
    cleaned_string = json_string.strip().replace("```json", "").replace("```", "")
    return cleaned_string.strip()

def split_into_chunks(df, cluster_col, target_chunk_size):
    """
    Split a DataFrame into smaller chunks based on cluster grouping.

    This function groups the DataFrame by the specified column and splits the data into chunks where each chunk
    does not exceed the target number of rows.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cluster_col (str): The column name to group by (e.g., 'assigned_cluster').
        target_chunk_size (int): Maximum number of rows allowed per chunk.

    Returns:
        list: A list of DataFrame chunks.
    """
    grouped = df.groupby(cluster_col)
    chunks = []
    current_chunk = pd.DataFrame()
    current_size = 0
    
    # Iterate over each group.
    for cluster, group in grouped:
        group_size = len(group)
        # If adding the group exceeds the target chunk size, save the current chunk and start a new one.
        if current_size + group_size > target_chunk_size:
            chunks.append(current_chunk)
            current_chunk = pd.DataFrame()
            current_size = 0
        
        # Append the group to the current chunk.
        current_chunk = pd.concat([current_chunk, group])
        current_size += group_size
    
    # Add the last chunk if it contains data.
    if not current_chunk.empty:
        chunks.append(current_chunk)
    
    return chunks

def process_cleaned_response(cleaned_response, classification_list):
    """
    Process the cleaned JSON response from GPT-4 and update the classification list.

    The function evaluates the cleaned JSON string into a Python object (dictionary or list of dictionaries)
    and validates that each entry is a dictionary. It then extends the classification_list with valid entries.

    Parameters:
        cleaned_response (str): The cleaned JSON string from GPT-4.
        classification_list (list): A list to which validated classification dictionaries will be added.

    Returns:
        None
    """
    print("Raw cleaned_response:", cleaned_response)
    
    # Convert the cleaned JSON string to a Python object.
    evaluated_response = ast.literal_eval(cleaned_response)
    print("Evaluated Response:", evaluated_response)
    
    # If the response is a single dictionary, wrap it in a list.
    if isinstance(evaluated_response, dict):
        validated_list = [evaluated_response]
    elif isinstance(evaluated_response, list):
        # Ensure every item in the list is a dictionary.
        for item in evaluated_response:
            if not isinstance(item, dict):
                print(f"item {item} not a dictionary")
                sys.exit(1)
        validated_list = [item for item in evaluated_response if isinstance(item, dict)]
    else:
        print(f"Error: Evaluated response {evaluated_response} is neither a list nor a dictionary.")
        validated_list = []
        sys.exit(1)
    
    # Add validated entries to the classification list.
    classification_list.extend(validated_list)

def classify_barrier_no_barrier(file_name):
    """
    Classify clusters of barriers by generating keyphrases and determining if each cluster contains a valid barrier.

    This function reads a CSV file containing initial clusters, generates a prompt for GPT-4 to produce keyphrases
    and a barrier classification (Barrier/Not a Barrier) for each cluster, processes the responses, and saves the
    results into separate CSV files for clusters with barriers and clusters without meaningful barriers.

    Parameters:
        file_name (str): Base file name used to construct input and output file paths.

    Returns:
        str: The file path of the CSV file containing clusters classified as having meaningful barriers.
    """
    input_file = f'./data/OpiatesRecovery_{file_name}_initial_clusters.csv'
    output_file = f'data/OpiatesRecovery_{file_name}_new_barriers_keyphrases.csv'
    output_file_no_barrier = f'data/OpiatesRecovery_{file_name}_new_barriers_keyphrases_no_barrier.csv'
    
    # Read the initial clusters CSV.
    df = pd.read_csv(input_file)
   
    # Prepare the system prompt for GPT-4.
    system_prompt = (
        "I have a list of barriers to opioid use disorder (OUD) recovery grouped by clusters. For each cluster of barriers "
        "provided below, generate two to three keyphrases that encapsulate the core semantic themes and underlying challenges "
        "related to opioid use disorder (OUD) recovery. The keyphrases should be: Abstract and Thematic: Focus on the main "
        "conceptual challenges rather than specific details. Generalized: Avoid mentioning specific substances, names, or overly "
        "specific scenarios unless they represent a fundamental aspect of the barrier. Consistent in Format: Use clear, concise noun "
        "phrases that can serve as categories for clustering. In addition, for each cluster, identify if the cluster contains "
        "meaningful barriers to recovery or not. If the cluster does not contain any meaningful barriers, return 'Not a Barrier'. "
        "Else return 'Barrier' under classification. Please return the results in the following JSON format:\n\n"
        "[{cluster_id: <cluster_id>, keyphrases: [<list of two to three keyphrases>], classification: <Barrier/Not a Barrier>}]\n\n"
        "Here are the barriers grouped by cluster:\n\n"
    )
    
    # Split the DataFrame into smaller chunks to avoid excessively large prompts.
    chunks = split_into_chunks(df, cluster_col="assigned_cluster", target_chunk_size=150)
    print("number of chunks:", len(chunks))
    classification_list = []

    # Process each chunk by generating prompts and retrieving GPT-4 responses.
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1} df size: {len(chunk)}")
        prompt = generate_clustered_keyphrase_prompt(chunk)
        
        messages_q1 = [
            {"role": "system", "content": "You are an expert in opioid use disorder recovery."},
            {"role": "user", "content": f'{system_prompt} {prompt}.\n'}
        ]
        
        response1 = get_gpt4_response(messages_q1, max_tokens=4096)
        print(f'response: {response1}')
        
        # Clean and process the GPT-4 response.
        cleaned_response = clean_json_string(response1)
        print(f'\ncleaned_response: {cleaned_response}')
        process_cleaned_response(cleaned_response, classification_list)

    print(classification_list)
    results_df = pd.DataFrame(classification_list)

    # Separate clusters based on the classification.
    filtered_df = results_df[results_df['classification'] != 'Not a Barrier']
    no_barrier_df = results_df[results_df['classification'] == 'Not a Barrier']

    # Save the results to CSV files.
    no_barrier_df.to_csv(output_file_no_barrier, index=False)
    filtered_df.to_csv(output_file, index=False)
    print(f'results saved to {output_file}')

    return output_file

# Example usage (uncomment to run as a standalone script)
# file_name = 'all_data'
# classified_df = classify_barrier_no_barrier(file_name)
