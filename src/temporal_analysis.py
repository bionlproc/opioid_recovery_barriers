import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

def split_dataframe_by_date(
    input_filepath,
    output_pre_pandemic,
    output_post_pandemic,
    date_column='created_utc',
    cutoff_date_str='2020-03-11',
    date_format=None
):
    """
    Splits an input CSV file into two CSV files based on a specified cutoff date.

    The input CSV is read into a DataFrame, the specified date column is converted to datetime,
    and the DataFrame is split into two parts: one containing rows up to the cutoff date (inclusive)
    and another with rows after the cutoff date.

    Parameters:
        input_filepath (str): Path to the input CSV file.
        output_pre_pandemic (str): Path where the pre-pandemic CSV file will be saved.
        output_post_pandemic (str): Path where the post-pandemic CSV file will be saved.
        date_column (str): The name of the column containing date information. Default is 'created_utc'.
        cutoff_date_str (str): The cutoff date in 'YYYY-MM-DD' format. Default is '2020-03-11'.
        date_format (str, optional): Format of the dates in the date_column. If None, pandas will infer the format.

    Returns:
        None
    """
    # Verify that the input file exists
    if not os.path.isfile(input_filepath):
        print(f"Error: The file '{input_filepath}' does not exist.")
        return

    # Read the input CSV into a DataFrame
    try:
        print(f"Reading the input file: {input_filepath}")
        df = pd.read_csv(input_filepath)
    except Exception as e:
        print(f"Error reading the file '{input_filepath}': {e}")
        return

    # Parse the date column into datetime format
    try:
        print(f"Parsing the '{date_column}' column to datetime.")
        df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='raise')
    except Exception as e:
        print(f"Error parsing dates in column '{date_column}': {e}")
        return

    # Define and print the cutoff date
    try:
        cutoff_date = pd.to_datetime(cutoff_date_str)
        print(f"Cutoff date set to: {cutoff_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error parsing the cutoff date '{cutoff_date_str}': {e}")
        return

    # Split the DataFrame into pre- and post-pandemic subsets
    print("Splitting the DataFrame based on the cutoff date...")
    pre_pandemic_df = df[df[date_column] <= cutoff_date].copy()
    post_pandemic_df = df[df[date_column] > cutoff_date].copy()

    # Display row counts for each subset
    print(f"Number of rows in pre-pandemic DataFrame (<= {cutoff_date_str}): {len(pre_pandemic_df)}")
    print(f"Number of rows in post-pandemic DataFrame (> {cutoff_date_str}): {len(post_pandemic_df)}")

    # Save the resulting DataFrames to CSV files
    try:
        print(f"Saving pre-pandemic data to '{output_pre_pandemic}'")
        pre_pandemic_df.to_csv(output_pre_pandemic, index=False)
    except Exception as e:
        print(f"Error writing to the file '{output_pre_pandemic}': {e}")
        return

    try:
        print(f"Saving post-pandemic data to '{output_post_pandemic}'")
        post_pandemic_df.to_csv(output_post_pandemic, index=False)
    except Exception as e:
        print(f"Error writing to the file '{output_post_pandemic}': {e}")
        return

    print("Data splitting completed successfully.")

def load_data(pre_pandemic_filepath, post_pandemic_filepath):
    """
    Loads pre-pandemic and post-pandemic CSV files into pandas DataFrames.
    
    Parameters:
        pre_pandemic_filepath (str): Path to the pre-pandemic CSV file.
        post_pandemic_filepath (str): Path to the post-pandemic CSV file.
    
    Returns:
        tuple: (df_pre, df_post) where each is a pandas DataFrame.
    """
    if not os.path.isfile(pre_pandemic_filepath):
        raise FileNotFoundError(f"Pre-pandemic file '{pre_pandemic_filepath}' not found.")
    if not os.path.isfile(post_pandemic_filepath):
        raise FileNotFoundError(f"Post-pandemic file '{post_pandemic_filepath}' not found.")
    
    df_pre = pd.read_csv(pre_pandemic_filepath)
    df_post = pd.read_csv(post_pandemic_filepath)
    
    print(f"Pre-pandemic data loaded: {df_pre.shape[0]} records.")
    print(f"Post-pandemic data loaded: {df_post.shape[0]} records.")
    
    return df_pre, df_post

def aggregate_barriers(df, category_column='lit_barrier_id'):
    """
    Aggregates barrier counts for each category.

    Parameters:
        df (pd.DataFrame): The DataFrame to aggregate.
        category_column (str): The column representing barrier categories.
    
    Returns:
        pd.DataFrame: A DataFrame with each category and its corresponding count.
    """
    df_agg = df.groupby(category_column).size().reset_index(name='count')
    return df_agg

def merge_aggregated_data(agg_pre, agg_post, category_column='lit_barrier_id'):
    """
    Merges pre- and post-pandemic aggregated barrier data on the specified category.

    Parameters:
        agg_pre (pd.DataFrame): Aggregated pre-pandemic DataFrame.
        agg_post (pd.DataFrame): Aggregated post-pandemic DataFrame.
        category_column (str): The column representing barrier categories.
    
    Returns:
        pd.DataFrame: A merged DataFrame containing pre and post counts for each category.
    """
    df_merged = pd.merge(
        agg_pre,
        agg_post,
        on=category_column,
        how='outer',
        suffixes=('_pre', '_post')
    ).fillna(0)
    
    # Convert count columns to integers
    df_merged['count_pre'] = df_merged['count_pre'].astype(int)
    df_merged['count_post'] = df_merged['count_post'].astype(int)
    
    return df_merged

def calculate_changes(df, pre_column='count_pre', post_column='count_post'):
    """
    Calculates absolute and percentage changes between pre- and post-pandemic barrier counts.

    Adds both raw and normalized change columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Merged DataFrame with pre and post counts.
        pre_column (str): Column name for pre-pandemic counts.
        post_column (str): Column name for post-pandemic counts.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for change metrics.
    """
    # Calculate absolute and percentage changes
    df['absolute_change'] = df[post_column] - df[pre_column]
    df['percentage_change'] = (df['absolute_change'] / df[pre_column].replace(0, pd.NA)) * 100
    df['percentage_change'] = df['percentage_change'].round(2)
    df['percentage_change'] = df['percentage_change'].fillna(0)

    # Normalize counts and compute changes on the normalized values
    df['count_pre_normalized'] = df[pre_column] / df[pre_column].sum()
    df['count_post_normalized'] = df[post_column] / df[post_column].sum()
    df['absolute_change_normalized'] = df['count_post_normalized'] - df['count_pre_normalized']
    df['percentage_change_normalized'] = (df['absolute_change_normalized'] / df['count_pre_normalized'].replace(0, pd.NA)) * 100
    df['percentage_change_normalized'] = df['percentage_change_normalized'].round(2)
    df['percentage_change_normalized'] = df['percentage_change_normalized'].fillna(0)
    
    return df

def calculate_normalized_changes(df, pre_column='count_pre', post_column='count_post'):
    """
    Calculates normalized absolute and percentage changes between pre- and post-pandemic counts.

    Parameters:
        df (pd.DataFrame): Merged DataFrame with pre and post counts.
        pre_column (str): Column name for pre-pandemic counts.
        post_column (str): Column name for post-pandemic counts.
    
    Returns:
        pd.DataFrame: DataFrame with normalized change columns.
    """
    df['count_pre_normalized'] = df[pre_column] / df[pre_column].sum()
    df['count_post_normalized'] = df[post_column] / df[post_column].sum()
    df['absolute_change_normalized'] = df['count_post_normalized'] - df['count_pre_normalized']
    df['percentage_change_normalized'] = (df['absolute_change_normalized'] / df[pre_column].replace(0, pd.NA)) * 100
    df['percentage_change_normalized'] = df['percentage_change_normalized'].round(2)
    df['percentage_change_normalized'] = df['percentage_change_normalized'].fillna(0)
    
    return df

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_barrier_counts(df, category_column='lit_barrier_id', pre_column='count_pre', post_column='count_post'):
    """
    Plots a side-by-side bar chart comparing pre- and post-pandemic barrier counts.

    Parameters:
        df (pd.DataFrame): DataFrame containing barrier counts.
        category_column (str): Column representing barrier categories.
        pre_column (str): Column name for pre-pandemic counts.
        post_column (str): Column name for post-pandemic counts.
    
    Returns:
        None
    """
    plt.figure(figsize=(30, 8))
    
    # Ensure categories are strings for proper labeling
    categories = df[category_column].astype(str)
    x = range(len(categories))
    bar_width = 0.55
    opacity = 0.8

    # Plot pre-pandemic counts
    plt.bar([i - bar_width/2 for i in x], df[pre_column], bar_width,
            alpha=opacity, label='Pre-Pandemic', color='skyblue')
    
    # Plot post-pandemic counts
    plt.bar([i + bar_width/2 for i in x], df[post_column], bar_width,
            alpha=opacity, label='Post-Pandemic', color='salmon')

    plt.xlabel(category_column, fontsize=10)
    plt.ylabel('Number of Barriers', fontsize=10)
    plt.title('Comparison of Barriers Pre and Post-Pandemic', fontsize=14)
    plt.xticks(ticks=x, labels=categories, rotation=90, ha='right', fontsize=6)
    plt.yticks(fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_percentage_change(df, category_column='lit_barrier_id', percentage_column='percentage_change'):
    """
    Plots a bar chart showing the percentage change in barrier counts per category.

    Parameters:
        df (pd.DataFrame): DataFrame containing percentage change data.
        category_column (str): Column representing barrier categories.
        percentage_column (str): Column representing percentage change.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_column, y=percentage_column, data=df, palette="viridis")
    plt.xlabel(category_column, fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
    plt.title('Percentage Change in Barriers Post-Pandemic')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_percentage_change_hue(df, category_column='lit_barrier_id', percentage_column='percentage_change'):
    """
    Plots a bar chart showing percentage changes with a gradient color scale based on the magnitude of change.

    Parameters:
        df (pd.DataFrame): DataFrame containing percentage change data.
        category_column (str): Column representing barrier categories.
        percentage_column (str): Column representing percentage change.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    
    # Split data into increasing and decreasing changes
    increasing_values = df[df[percentage_column] > 0].sort_values(by=percentage_column)
    decreasing_values = df[df[percentage_column] < 0].sort_values(by=percentage_column, ascending=False)

    # Create colormap gradients for increasing and decreasing values
    reds = plt.cm.Reds(np.linspace(0.3, 1, len(increasing_values)))
    greens = plt.cm.Greens(np.linspace(0.3, 1, len(decreasing_values)))

    # Map category names to colors
    colors = {}
    for idx, val in enumerate(increasing_values[percentage_column]):
        colors[increasing_values[category_column].iloc[idx]] = mcolors.to_hex(reds[idx])
    for idx, val in enumerate(decreasing_values[percentage_column]):
        colors[decreasing_values[category_column].iloc[idx]] = mcolors.to_hex(greens[idx])

    # Assign colors to the DataFrame based on the category
    df['color'] = df[category_column].map(colors)
    
    # Plot using the assigned colors
    sns.barplot(
        x=category_column,
        y=percentage_column,
        data=df,
        palette=df['color'].tolist()
    )
    
    plt.xlabel(category_column, fontsize=14)
    plt.ylabel('Percentage Change (%)', fontsize=14)
    plt.title('Percentage Change in Barriers Post-Pandemic', fontsize=15)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./data/images/temporal_changes_pre_post_covid_.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()

def analyze_oud_trends(pre_pandemic_csv, post_pandemic_csv, data_analysis_csv, category_column='lit_barrier_id'):
    """
    Analyzes opioid use disorder (OUD) trends by comparing pre- and post-pandemic barrier data.

    Loads pre- and post-pandemic datasets, aggregates barrier counts by category, merges the data,
    calculates changes, visualizes the results, and saves the final analysis to a CSV file.

    Parameters:
        pre_pandemic_csv (str): Path to the pre-pandemic CSV file.
        post_pandemic_csv (str): Path to the post-pandemic CSV file.
        data_analysis_csv (str): Path where the final analysis CSV will be saved.
        category_column (str): Column representing barrier categories.
    
    Returns:
        None
    """
    # Load data
    df_pre, df_post = load_data(pre_pandemic_csv, post_pandemic_csv)

    # Aggregate barrier counts for pre and post datasets
    agg_pre = aggregate_barriers(df_pre, category_column)
    agg_post = aggregate_barriers(df_post, category_column)

    print("Pre-Pandemic Barrier Counts:")
    print(agg_pre)
    print("\nPost-Pandemic Barrier Counts:")
    print(agg_post)

    # Merge aggregated counts
    df_combined = merge_aggregated_data(agg_pre, agg_post, category_column)
    print("\nCombined Barrier Counts:")
    print(df_combined)

    # Calculate changes
    df_changes = calculate_changes(df_combined)
    print("\nBarrier Changes:")
    print(df_changes)

    # Visualize the barrier counts and percentage changes
    plot_barrier_counts(df_changes, category_column=category_column)
    
    if category_column == 'lit_barrier_id':
        plot_percentage_change_hue(df_changes, category_column=category_column)
        plot_percentage_change_hue(df_changes, category_column=category_column, percentage_column='percentage_change_normalized')
    else:
        plot_percentage_change(df_changes, category_column=category_column)
        plot_percentage_change(df_changes, category_column=category_column, percentage_column='percentage_change_normalized')

    # Save the final analysis to CSV
    df_changes.to_csv(data_analysis_csv, index=False)

# =============================================================================
# Main Execution Block 
# =============================================================================

if __name__ == "__main__":
    # Define file paths for temporal analysis of barriers
    pre_pandemic_lit =  "./data/temporal/opioid_recovery_barriers_pre_pandemic_lit.csv" 
    post_pandemic_lit =  "./data/temporal/opioid_recovery_barriers_post_pandemic_lit.csv" 
    data_analysis_lit =  "./data/temporal/opioid_recovery_barriers_pre_post_pandemic_trends_lit.csv" 

    pre_pandemic_new = "./data/temporal/opioid_recovery_barriers_pre_pandemic_new_barriers.csv" 
    post_pandemic_new = "./data/temporal/opioid_recovery_barriers_post_pandemic_new_barriers.csv" 
    data_analysis_new =  "./data/temporal/opioid_recovery_barriers_pre_post_pandemic_trends_new_barriers.csv" 

    # Input CSV files with dates for barriers
    new_barriers_with_date_csv = "./data/OpiatesRecovery_new_barriers_date.csv"  
    lit_barriers_with_date_csv = "./data/OpiatesRecovery_lit_barriers_date.csv"  

    # Split data into pre and post pandemic datasets
    split_dataframe_by_date(lit_barriers_with_date_csv, pre_pandemic_lit, post_pandemic_lit)
    split_dataframe_by_date(new_barriers_with_date_csv, pre_pandemic_new, post_pandemic_new)

    # Analyze trends for literature-based barriers
    analyze_oud_trends(pre_pandemic_lit, post_pandemic_lit, data_analysis_lit)
    
    # Analyze trends for new barriers (using 'Rank' as the category column)
    analyze_oud_trends(pre_pandemic_new, post_pandemic_new, data_analysis_new, 'Rank')
