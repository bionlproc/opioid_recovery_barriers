import time
from tqdm import tqdm
from utility import read_posts, get_gpt4_response

# Initialize the tqdm extension for pandas to allow progress_apply on DataFrame columns
tqdm.pandas()


def get_barriers_and_verify_with_chat(post: str) -> tuple:
    """
    Extract and verify barriers to recovery from an opioid use disorder Reddit post.

    This function uses a two-step conversation with GPT-4:
      1. Extraction: It instructs GPT-4 to extract the barriers explicitly mentioned by the user.
      2. Verification: It then asks GPT-4 to verify that the identified barriers are explicitly mentioned or strongly indicated as contributing to relapse.

    Parameters:
        post (str): The Reddit post content.

    Returns:
        tuple: A tuple containing:
            - response_extraction (str): The initial GPT-4 response with extracted barriers.
            - response_verification (str): The GPT-4 response after verifying the extracted barriers.
    """
    # Step 1: Set up the conversation context and instruct GPT-4 to extract barriers.
    messages = [
        {"role": "system", "content": "You are a skilled information extraction agent."},
        {"role": "user", "content": f"""You are given a Reddit post. Your task is to extract barriers to recovery from opioid use disorder as explicitly mentioned by the user. Strictly adhere to the following guidelines when extracting the barriers:

        The user is talking about their own experience and not someone else’s.
        The barrier is explicitly mentioned by the user or has strong indications as causing them to relapse or contributing to the risk of relapse. Discard barriers that do not adhere to the above guidelines.
        If no barriers are found, mention "No barriers found." Only use the details provided by the user in the post, without relying on previous knowledge on the subject or making assumptions.

        Provide reasons for the selection of the items.

        Finally, provide the items as a numbered list as follows:
        Identified barriers:

        <barrier 1>
        <barrier 2>
         ...\n\nPost:
         {post}"""}
    ]
    # Make the first API call to extract the barriers from the post.
    response_extraction = get_gpt4_response(messages)
    print("Initial Barriers:\n", response_extraction)

    # Step 2: Append the extraction result to the conversation and request verification.
    messages.append({"role": "assistant", "content": response_extraction})
    messages.append({
        "role": "user",
        "content": "Verify that the user explicitly mentions or has strong indications of the identified items as causing or contributing to relapse or shows strong indications of presenting challenges in maintaining recovery. The user must be talking about their own recovery."
    })
    # Make the second API call for verification within the same conversation.
    response_verification = get_gpt4_response(messages)
    time.sleep(5)  # Pause briefly to manage API rate limits or processing time.
    print("Verified Barriers:\n", response_verification)

    return response_extraction, response_verification


def identify_finalized_barriers(verified_list_of_barriers: str) -> str:
    """
    Finalize the list of barriers by refining the verified output.

    This function sends the verified barriers to GPT-4 for final formatting. It ensures that each barrier is 
    presented in a numbered list with clear, detailed descriptions. If no barriers are found, it returns "No barriers found."

    Parameters:
        verified_list_of_barriers (str): Text containing the verified barriers extracted from the post.

    Returns:
        str: The finalized, formatted list of barriers.
    """
    # Create a conversation with detailed instructions for formatting the finalized barriers.
    messages_q1 = [
        {"role": "system", "content": "You are a skilled information extraction agent."},
        {"role": "user", "content": f'You are given information about potential barriers to recovery as mentioned by Reddit users in their posts, along with a verified list of barriers. Your task is to extract the finalized list of barriers from the provided text. Ensure each barrier is represented as a numbered list using clear and meaningful sentences that accurately capture the context and details without shortening them excessively. The barriers should be concise yet detailed enough for someone reviewing them later to fully understand what each barrier entails. If no barriers are found, return "No barriers found." \n\nInfo on barriers to recovery:\n"{verified_list_of_barriers}\nList of barriers to recovery:\n'}
    ]
    # Request GPT-4 to format the verified barriers.
    barriers_answer = get_gpt4_response(messages_q1)
    print('Finalized list of barriers:\n', barriers_answer)
    time.sleep(10)  # Delay to manage processing constraints.
    return barriers_answer


def extract_finalized_barriers(input_file: str, output_file: str) -> None:
    """
    Read Reddit posts, extract and verify recovery barriers, and save the finalized results.

    This function performs the following steps:
      1. Reads posts from a CSV file.
      2. Applies the barrier extraction and verification process to each post.
      3. Finalizes the verified barriers into a refined list.
      4. Saves the updated DataFrame with new columns to an output CSV file.

    Parameters:
        input_file (str): Path to the CSV file containing Reddit posts.
        output_file (str): Path where the processed CSV file with finalized barriers will be saved.
    """
    # Load posts into a DataFrame.
    posts_df = read_posts(input_file)
    
    # Extract potential and verified barriers from each cleaned post.
    posts_df['potential_barriers_to_recovery'], posts_df['verified_barriers_to_recovery'] = zip(
        *posts_df['post_cleaned'].progress_apply(get_barriers_and_verify_with_chat)
    )
    
    # Refine the verified barriers into a finalized list.
    posts_df['barriers_to_recovery_filtered'] = posts_df['verified_barriers_to_recovery'].progress_apply(identify_finalized_barriers)
    
    # Save the updated DataFrame to a CSV file.
    posts_df.to_csv(output_file, index=False)


# =============================================================================
# Main Execution Block
# =============================================================================
# if __name__ == "__main__":
#     # Define the input and output CSV file paths.
#     input_file = './data/OpiatesRecovery_code_test_posts.csv'
#     output_file = './data/OpiatesRecovery_code_test_posts_finalized_barriers.csv'

#     # Execute the process to extract and finalize barriers from the posts.
#     extract_finalized_barriers(input_file, output_file)
