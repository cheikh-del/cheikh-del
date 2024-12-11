import os
import pandas as pd
from datetime import datetime
from pubmed_pipeline.fetch_pubmed_articles import fetch_pubmed_articles_by_week
from pubmed_pipeline.process_bionlp import process_compiled_file_with_bionlp
from pubmed_pipeline.compute_entity_cooccurrences import compute_entity_cooccurrences
from pubmed_pipeline.compute_entity_similarities import compute_entity_similarities
from pubmed_pipeline.ngram_frequencies_extraction import extract_ngrams_table, merge_ngrams_files
from itertools import combinations

# === Configuration des chemins ===
BASE_DIR = "C:/collecte/biolm/Test"
input_dir = os.path.join(BASE_DIR, "Input")
output_dir = os.path.join(BASE_DIR, "Output")
entities_dir = os.path.join(output_dir, "entities")
cooccurrences_dir = os.path.join(output_dir, "cooccurrences")
similarities_dir = os.path.join(output_dir, "similarities")
ngrams_dir = os.path.join(output_dir, "ngrams")

for directory in [input_dir, output_dir, entities_dir, cooccurrences_dir, similarities_dir, ngrams_dir]:
    os.makedirs(directory, exist_ok=True)

# === Pipeline ===
def run_pipeline():
    print("[INFO] Starting pipeline...")

    # Étape 1 : Collecte des articles PubMed
    search_term = "biomarkers"
    start_date = datetime(1946, 1, 1)
    end_date = datetime(2024, 12, 31)
    try:
        if not any(file.endswith(".csv") for file in os.listdir(input_dir)):
            print("[INFO] Fetching PubMed articles...")
            fetch_pubmed_articles_by_week(search_term, start_date, end_date, batch_size=1000, output_directory=input_dir)
            print("[SUCCESS] PubMed articles fetched successfully.")
        else:
            print("[INFO] PubMed articles already fetched. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch PubMed articles: {e}")
        return

    # Étape 2 : Extraction des entités
    entities_file = os.path.join(entities_dir, "pubmed_cosmetics_entities_all.csv")
    try:
        if not os.path.exists(entities_file):
            print("[INFO] Extracting entities...")
            process_compiled_file_with_bionlp(input_dir, entities_dir)
            if not os.path.exists(entities_file):
                raise FileNotFoundError(f"[ERROR] Entities file not created at {entities_file}")
            print("[SUCCESS] Entities extracted successfully.")
        else:
            print("[INFO] Entities already extracted. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to extract entities: {e}")
        return

    # Étape 3 : Création du corpus
    corpus_file = os.path.join(cooccurrences_dir, "corpus.csv")
    try:
        if not os.path.exists(corpus_file):
            print("[INFO] Creating corpus...")
            corpus_file = create_corpus(entities_file, cooccurrences_dir)
            if not corpus_file:
                raise ValueError("[ERROR] Corpus creation failed.")
            print("[SUCCESS] Corpus created successfully.")
        else:
            print("[INFO] Corpus already created. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to create corpus: {e}")
        return

    # Étape 4 : Calcul des cooccurrences
    cooccurrence_file = os.path.join(cooccurrences_dir, "entity_cooccurrences.csv")
    try:
        if not os.path.exists(cooccurrence_file):
            print("[INFO] Calculating cooccurrences...")
            cooccurrence_file = compute_entity_cooccurrences(corpus_file, cooccurrences_dir)
            if not cooccurrence_file:
                raise ValueError("[ERROR] Cooccurrence calculation failed.")
            print("[SUCCESS] Cooccurrences calculated successfully.")
        else:
            print("[INFO] Cooccurrences already calculated. Skipping step.")
    except Exception as e:
        print(f"[ERROR] Failed to compute cooccurrences: {e}")
        return

    print("[SUCCESS] Pipeline completed!")

# === Création du corpus ===
def create_corpus(input_file, output_directory):
    try:
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip().str.upper()

        # Nettoyage des données
        df = df.dropna(subset=["ENTITY", "LABEL", "PUBMED_ID"]).drop_duplicates()
        df["ENTITY"] = df["ENTITY"].apply(lambda x: str(x).strip())
        df["LABEL"] = df["LABEL"].apply(lambda x: str(x).strip())

        # Création du corpus
        corpus = df.groupby("PUBMED_ID")[["ENTITY", "LABEL"]].apply(
            lambda x: list(zip(x["ENTITY"], x["LABEL"]))
        ).reset_index(name="DOCUMENT")

        # Sauvegarde
        os.makedirs(output_directory, exist_ok=True)
        corpus_file = os.path.join(output_directory, "corpus.csv")
        corpus.to_csv(corpus_file, index=False)
        return corpus_file
    except Exception as e:
        print(f"[ERROR] Failed to create corpus: {e}")
        return None

# === Exécution ===
if __name__ == "__main__":
    run_pipeline()
