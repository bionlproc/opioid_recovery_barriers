import time
import numpy as np
import pandas as pd
from openai import OpenAI

def load_api_key(filepath):
    """
    Load the API key from a text file.
    
    Parameters:
        filepath (str): The path to the file containing the API key.
    
    Returns:
        str: The API key.
    """
    try:
        with open(filepath, 'r') as f:
            api_key = f.read().strip()
        return api_key
    except Exception as e:
        print(f"Error loading API key from {filepath}: {e}")
        return None

# =============================================================================
# API Initialization
# =============================================================================
# Load the OpenAI API key from a file.
api_key_file = './data/api_keys/api_key.txt'  # Ensure this file exists and contains your API key.
openai_api_key = load_api_key(api_key_file)
if openai_api_key is None:
    raise ValueError("API key could not be loaded. Please check the api_key file.")

client = OpenAI(api_key=openai_api_key)

def read_posts(path):
    """
    Reads a CSV file containing posts and returns a DataFrame.
    
    Parameters:
        path (str): File path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    df = pd.read_csv(path)
    return df

# =============================================================================
# OpenAI GPT-4 API Wrapper Functions
# =============================================================================
def get_gpt4_response(prompt, temp=0.3, max_tokens=1024):
    """
    Sends a chat-based prompt to GPT-4 and returns the generated response.

    Parameters:
        prompt (list): A list of message dictionaries (in the Chat API format).
        temp (float, optional): Temperature setting for response variability. Default is 0.3.
        max_tokens (int, optional): Maximum number of tokens to generate. Default is 1024.

    Returns:
        str or None: The generated response text from GPT-4, or None if an error occurs.
    """
    try:
        # Create a GPT-4 chat completion
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=prompt,
            max_tokens=max_tokens,  # Allow longer responses if needed.
            n=1,
            stop=None,
            temperature=temp,
        )
        # Return the generated content
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred while processing the request: {e}")
        return None 

def get_embedding(text, model="text-embedding-3-large"):
    """
    Retrieve an embedding vector for the given text using OpenAI's embeddings API.

    Parameters:
        text (str): The text input for which to generate an embedding.
        model (str, optional): The embedding model to use. Default is "text-embedding-3-large".

    Returns:
        list: A list of floats representing the embedding vector.
    """
    response = client.embeddings.create(input=text, model=model)
    time.sleep(1)  # Pause briefly to respect rate limits.
    return response.data[0].embedding

def load_mean_embeddings(npz_file_path):
    """
    Loads mean embeddings from an .npz file.

    The .npz file is expected to contain keys mapping cluster IDs to their corresponding mean embeddings.

    Parameters:
        npz_file_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary mapping cluster IDs (as strings) to their mean embedding vectors.
    """
    try:
        # Load the .npz file containing mean embeddings.
        loaded_data = np.load(npz_file_path)
        # Convert the NpzFile object to a regular dictionary.
        mean_embeddings_dict = {key: loaded_data[key] for key in loaded_data.files}
        print(f"Loaded mean embeddings for {len(mean_embeddings_dict)} clusters from '{npz_file_path}'.")
        return mean_embeddings_dict
    except FileNotFoundError:
        print(f"Error: The file '{npz_file_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading the .npz file: {e}")
        return {}

def load_literature_barrier_embeddings(lit_input_file):
    """
    Loads literature barrier embeddings and their corresponding labels from an .npz file.

    Parameters:
        lit_input_file (str): Path to the literature .npz file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The barrier literature embeddings.
            - np.ndarray: The barrier literature labels.
        Returns (None, None) if an error occurs.
    """
    try:
        lit_data = np.load(lit_input_file)
        barrier_lit_embeddings = lit_data['barrier_lit_embeddings']
        barrier_lit_labels = lit_data['barriers_lit']
        print(f"Loaded literature barrier embeddings and labels from '{lit_input_file}'.")
        return barrier_lit_embeddings, barrier_lit_labels
    except FileNotFoundError:
        print(f"Error: The file '{lit_input_file}' was not found.")
        return None, None
    except KeyError as e:
        print(f"Error: Missing key in the literature .npz file - {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading '{lit_input_file}': {e}")
        return None, None
