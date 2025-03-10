import pandas as pd
from utility import read_posts
import re

def is_potential_barrier(text):
    """
    Determine whether a given text is a potential barrier.

    This function checks the text against common patterns that indicate non-barrier statements.
    If the text matches any of these patterns, it is not considered a potential barrier.

    Parameters:
        text (str): The text to evaluate.

    Returns:
        bool: True if the text is a potential barrier, False otherwise.
    """
    # Define patterns that signal non-barrier statements.
    non_barrier_patterns = [
        r"^Based on the.*",          # E.g., "Based on the provided text"
        r"No barriers found.*",      # E.g., "No barriers found."
        r"^According to.*",          # Generic informational text
        r"^In the context of.*"       # Irrelevant context
    ]
    
    # Check if the text matches any non-barrier pattern.
    for pattern in non_barrier_patterns:
        if re.match(pattern, text):
            return False
    return True

def split_and_clean_barriers(row):
    """
    Split and clean barrier text from a DataFrame row.

    This function processes the 'barriers_to_recovery_filtered' column by splitting the text on newline characters,
    removing any numbering or leading punctuation, and filtering out non-barrier statements. If the only barrier is
    "No barriers found.", the function returns None.

    Parameters:
        row (pd.Series): A DataFrame row containing at least 'id' and 'barriers_to_recovery_filtered'.

    Returns:
        pd.DataFrame or None: A DataFrame with columns 'id' and 'barrier' for each cleaned barrier,
                              or None if no valid barriers are found.
    """
    barriers = row['barriers_to_recovery_filtered']
    
    if pd.isna(barriers):  # Skip if the barrier text is missing.
        print("ERROR: nan found for barriers")
        return None
    
    # Split the text into individual barrier entries.
    split_barriers = barriers.split("\n")
    
    # Remove leading numbers and punctuation (e.g., "1. ", "2. ") and extra spaces.
    clean_barriers = [barrier.lstrip('0123456789. ').strip() for barrier in split_barriers if barrier.strip()]

    # If the only barrier is "No barriers found.", skip this row.
    if len(clean_barriers) == 1 and clean_barriers[0].lower() == 'no barriers found.':
        print("\nclean barriers", clean_barriers)
        return None
    
    # Create a DataFrame with each cleaned barrier associated with the row's 'id'.
    barriers_df = pd.DataFrame({'id': [row['id']] * len(clean_barriers), 'barrier': clean_barriers})
    
    # Filter out entries that are not considered potential barriers.
    barriers_df = barriers_df[barriers_df['barrier'].apply(is_potential_barrier)]

    return barriers_df

def generate_barrier_df(input_file, output_file):
    """
    Generate a DataFrame of potential barriers from posts and save to a CSV file.

    This function reads posts from the input file, extracts and cleans barrier text from each row,
    concatenates the results into a single DataFrame, and saves the final DataFrame to the output file.

    Parameters:
        input_file (str): Path to the input file containing posts.
        output_file (str): Path to the CSV file where the final barrier DataFrame will be saved.
    """
    result_dfs = []
    data_df = read_posts(input_file)
    
    # Process each row in the DataFrame to extract potential barriers.
    for index, row in data_df.iterrows():
        result = split_and_clean_barriers(row)
        if result is not None:
            result_dfs.append(result)

    # Concatenate all individual DataFrames into one final DataFrame.
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    # Save the final DataFrame to a CSV file.
    final_df.to_csv(output_file, index=False)

    # Print summary statistics.
    print(f"Number of unique IDs (Original df): {data_df['id'].nunique()}")
    unique_id_count = final_df['id'].nunique()
    print(f"Number of unique IDs (Final df): {unique_id_count}")


# Example execution block (uncomment to run as a standalone script)
# if __name__ == "__main__":
#     input_file = './data/OpiatesRecovery_code_test_posts_finalized_barriers.csv'
#     output_file = './data/OpiatesRecovery_code_test_posts_barriers_list.csv'
#     generate_barrier_df(input_file, output_file)
