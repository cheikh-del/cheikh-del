import os
from datetime import datetime
from fetch_pubmed_articles import fetch_pubmed_articles_by_week
from process_bionlp import process_compiled_file_with_bionlp
from compute_entity_cooccurrences import compute_entity_cooccurrences
from compute_entity_similarities import compute_entity_similarities
from ngram_frequencies_extraction import extract_ngrams_table, merge_ngrams_files

def run_pipeline():
    print("[INFO] Starting the PubMed pipeline...")

    # Configuration des chemins
    BASE_DIR = "/content/drive/MyDrive/pubmed_pipeline"
    input_dir = os.path.join(BASE_DIR, "input")
    output_dir = os.path.join(BASE_DIR, "output")
    entities_dir = os.path.join(output_dir, "entities")
    cooccurrences_dir = os.path.join(output_dir, "cooccurrences")
    similarities_dir = os.path.join(output_dir, "similarities")
    ngrams_dir = os.path.join(output_dir, "ngrams")

    # Création des répertoires si nécessaire
    for directory in [input_dir, output_dir, entities_dir, cooccurrences_dir, similarities_dir, ngrams_dir]:
        os.makedirs(directory, exist_ok=True)

    # Étape 1 : Collecte des articles
    try:
        csv_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]
        if not csv_files:
            print("[INFO] Fetching PubMed articles...")
            fetch_pubmed_articles_by_week(
                search_term="cosmetics",
                start_date=datetime.strptime("1946-01-01", "%Y-%m-%d").date(),
                end_date=datetime.strptime("2024-12-31", "%Y-%m-%d").date(),
                batch_size=1000,
                output_directory=input_dir
            )
            print("[SUCCESS] Articles fetched successfully.")
        else:
            print("[INFO] Articles already fetched. Skipping fetch step.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch articles: {e}")
        return

    # Étape 2 : Extraction des entités
    try:
        entities_file = os.path.join(entities_dir, "entities.csv")
        if not os.path.exists(entities_file):
            print("[INFO] Extracting entities...")
            process_compiled_file_with_bionlp(input_dir, entities_dir)
            print("[SUCCESS] Entities extracted successfully.")
        else:
            print("[INFO] Entities already extracted. Skipping extraction step.")
    except Exception as e:
        print(f"[ERROR] Failed to extract entities: {e}")
        return

    # Étape 3 : Calcul des cooccurrences
    try:
        cooccurrence_file = os.path.join(cooccurrences_dir, "entity_cooccurrences.csv")
        if not os.path.exists(cooccurrence_file):
            print("[INFO] Calculating entity cooccurrences...")
            compute_entity_cooccurrences(entities_file, cooccurrences_dir)
            print("[SUCCESS] Entity cooccurrences calculated successfully.")
        else:
            print("[INFO] Cooccurrences already calculated. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to calculate cooccurrences: {e}")
        return

    # Étape 4 : Calcul des similarités
    try:
        similarity_file = os.path.join(similarities_dir, "entity_similarities.csv")
        if not os.path.exists(similarity_file):
            print("[INFO] Calculating entity similarities...")
            compute_entity_similarities(cooccurrence_file, similarities_dir)
            print("[SUCCESS] Entity similarities calculated successfully.")
        else:
            print("[INFO] Similarities already calculated. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to calculate similarities: {e}")
        return

    # Étape 5 : Extraction des N-grams
    try:
        ngrams_output_file = os.path.join(ngrams_dir, "ngrams.csv")
        if not os.path.exists(ngrams_output_file):
            print("[INFO] Extracting and merging N-grams...")
            merge_ngrams_files(cooccurrences_dir, ngrams_output_file)
            print("[SUCCESS] N-grams extracted and merged successfully.")
        else:
            print("[INFO] N-grams already extracted and merged. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to extract and merge N-grams: {e}")
        return

    print("[SUCCESS] Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()
