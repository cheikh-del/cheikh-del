import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams_table(texts, ngram_range=(1, 3)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    ngrams = vectorizer.fit_transform(texts)
    ngrams_list = vectorizer.get_feature_names_out()
    ngram_counts = ngrams.toarray().sum(axis=0)

    ngram_table = pd.DataFrame({
        "N": [len(ngram.split()) for ngram in ngrams_list],
        "Ngram": ngrams_list,
        "Count": ngram_counts
    }).sort_values(by="Count", ascending=False)
    return ngram_table

def merge_ngrams_files(input_directory, output_file):
    """
    Merge all N-gram CSV files in a directory into a single file.

    Parameters:
        input_directory (str): Directory containing the N-gram CSV files.
        output_file (str): Path to save the merged N-gram file.

    Returns:
        None
    """
    # List all CSV files in the directory
    ngram_files = [f for f in os.listdir(input_directory) if f.endswith('_ngrams.csv')]
    
    if not ngram_files:
        print("[WARNING] No N-grams files found in the specified directory.")
        return

    print(f"[INFO] Found {len(ngram_files)} N-grams files. Merging...")

    # Initialize an empty DataFrame for merging
    merged_df = pd.DataFrame()

    # Iterate over all files and merge them
    for file in ngram_files:
        file_path = os.path.join(input_directory, file)
        print(f"[INFO] Processing file: {file_path}")
        
        # Load the N-gram CSV file
        df = pd.read_csv(file_path)
        
        if 'N' not in df.columns or 'Ngram' not in df.columns or 'Count' not in df.columns:
            print(f"[WARNING] File {file} does not have the required columns. Skipping.")
            continue

        # Merge with the main DataFrame
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Group by 'N' and 'Ngram' to aggregate the counts and merge the sentences
    merged_df = (
        merged_df.groupby(['N', 'Ngram'], as_index=False)
        .agg({
            'Count': 'sum',  # Sum the counts for identical N-grams
            'Sentences': lambda x: list(set(sum(x.dropna().apply(eval).tolist(), [])))  # Combine sentences without duplicates
        })
        .sort_values(by="Count", ascending=False)
    )

    # Save the merged DataFrame to a single CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Merged N-grams file saved to: {output_file}")
