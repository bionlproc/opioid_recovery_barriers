# Opioid Recovery Barriers Analysis

This project provides a pipeline for extracting and analyzing barriers to opioid recovery. It extracts, cleans, and processes textual data from Reddit posts (subreddit r/OpiatesRecovery). Then classifies and clusters barriers, and performs temporal trend analysis (e.g., pre- and post-pandemic). The pipeline uses OpenAI's GPT-4 API along with popular libraries such as Pandas, NumPy, and scikit-learn.

## Project Structure
```
opioid_recovery_barriers/    
            ├── data/                                      # Directory for input and output data files
            │   ├── api_keys/ 
            │   |   ├── api_key.txt                        # File containing the OpenAI API key
            │   ├── embeddings/                            # Stores generated embeddings in .npz format
            │   ├── temporal/                              # Stores temporal analysis results
            │   ├── images/                                # Stores visualization plots
            │   ├── barriers_to_recovery_lit.txt           # Stores the literature derived barriers
            │   ├── OpiatesRecovery_code_test_posts.csv    # Sample data file
            │   └── ... (other CSV files)
            ├── src/                                       # Source code directory for Python scripts
            │   ├── main.py                                # Main pipeline: extraction, embedding, classification, clustering, and descriptor generation
            │   ├── extract_barriers.py                    # Functions for extracting and finalizing barriers from raw posts
            │   ├── generate_embeddings.py                 # Functions to generate embeddings (barriers, keyphrases, etc.)
            │   ├── classification.py                      # Functions for classifying barriers using literature embeddings
            │   ├── barriers_data_cleaning.py              # Functions for cleaning and splitting barrier data into individual entries
            │   ├── clustering.py                          # Functions for initial and secondary clustering of barrier embeddings
            │   ├── generate_keyphrases.py                 # Functions to classify clusters with keyphrases and filter out non-barriers
            │   ├── generate_cluster_descriptors.py        # Functions to generate human-readable cluster descriptors from clusters
            │   ├── temporal_analysis.py                   # Temporal analysis: data splitting, aggregation, change calculations, and visualizations
            │   └── utils.py                               # Utility functions 
            ├── README.md                                  # Project documentation file
            ├── requirements.txt                           # List of dependencies required to run the project

```

## Setup

#### 1. Clone the repository:

```git clone https://github.com/vinuekanayake/opioid_recovery_barriers.git```

``` cd opioid_recovery_barriers```


#### 2. Install the dependencies:

```pip install -r requirements.txt```

#### 3. Configure your API key:

Create a directory named ```api_keys/``` in the ```data/``` directory.
Inside ```api_keys/```, create a file named ```api_key.txt```.
Paste your OpenAI API key inside api_key and save.

## Usage
### Running the Main Pipeline

To run the main pipeline, execute:

```python src/main.py```

The main pipeline performs the following steps:

**1.Extract Finalized Barriers:** Processes input posts and extracts barriers to recovery in a multi-step process.

**2.Generate Individual Barriers DataFrame:** Cleans and splits finalized barriers into individual entries.

**3.Generate Literature Barrier Embeddings:** Generates embeddings from ```data/barriers_to_recovery_lit.txt ``` file of literature derived barriers.

**4.Generate Barrier Embeddings from Posts:** Creates embeddings for each individual barrier.

**5.Classify Barriers:** Compares barrier embeddings with literature embeddings to classify them.

**6.Initial Clustering:** Clusters the classified barrier embeddings using agglomerative hierarchical clustering.

**7.Generate Keyphrases and Filter Clusters:** Generate keyphrases for the initial clusters generated in ```6.```. Filters clusters to remove those without meaningful barriers.

**8.Generate Keyphrase Embeddings:** Computes keyphrase embeddings for the keyphrases generated for each cluster.

**9.Secondary Clustering:** Refines clusters by combining barrier and keyphrase embeddings.

**10.Generate Cluster Descriptors:** Produces concise, human-readable descriptors for each cluster.


### Running Temporal Analysis

To run the temporal analysis module, execute:

```python src/temporal_analysis.py```

The `temporal_analysis.py` module performs the following:

- Splits input CSV files into pre- and post-pandemic datasets based on a specified cutoff date  
  - default:`2020-03-11` The cutoff date is in `YYYY-MM-DD` format.  
- Aggregates barrier counts by category.  
- Calculates both absolute and percentage changes between pre- and post-pandemic periods.  
- Visualizes trends using bar charts.


