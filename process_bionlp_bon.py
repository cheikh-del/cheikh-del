import os
import spacy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

def process_compiled_file_with_bionlp(file_path, output_directory, batch_size=1000, n_jobs=-1):
    """
    Process a PubMed articles CSV file with the BioNLP model and extract entities in parallel.
    """
    try:
        nlp_bionlp = spacy.load("en_ner_bionlp13cg_md", disable=["parser", "tagger", "lemmatizer"])
        print("[INFO] BioNLP model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading BioNLP model: {e}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] File {file_path} loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return

    df.columns = df.columns.str.strip().str.upper()

    # Vérification des colonnes requises
    required_columns = {'PUBMED_ID', 'TITLE', 'ABSTRACT', 'PUBLICATION DATE'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"[ERROR] Missing columns in file {file_path}: {', '.join(missing_columns)}. Skipping file.")
        return

    # Filtrage des lignes avec des titres ou résumés manquants
    df = df.dropna(subset=['TITLE', 'ABSTRACT'])
    total_rows = len(df)

    if total_rows == 0:
        print(f"[WARNING] File {file_path} contains no valid rows after cleaning. Skipping processing.")
        return

    print("[INFO] Processing articles in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(nlp_bionlp, row)
        for _, row in tqdm(df.iterrows(), total=total_rows, desc="Processing articles")
    )

    # Flatten results and filter empty
    results = [item for sublist in results if sublist for item in sublist]

    if not results:
        print(f"[WARNING] No entities extracted from file {file_path}. Skipping output.")
        return

    # Sauvegarde des entités extraites
    entities_df = pd.DataFrame(results)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_directory, f"{base_name}_entities.csv")
    try:
        os.makedirs(output_directory, exist_ok=True)
        entities_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] All extracted entities saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save entities to {output_file}: {e}")

def process_row(nlp_bionlp, row):
    pubmed_id = row.get('PUBMED_ID', 'Unknown')
    title = row.get('TITLE', '').strip()
    abstract = row.get('ABSTRACT', '').strip()
    year = row.get('PUBLICATION DATE', 'Unknown')

    if not title or not abstract:
        return []

    return extract_entities_bionlp(nlp_bionlp, pubmed_id, title, abstract, year)

def extract_entities_bionlp(nlp_bionlp, pubmed_id, title, abstract, year):
    entities = []
    try:
        doc = nlp_bionlp(f"{title} {abstract}")
        seen_entities = set()
        for ent in doc.ents:
            normalized_entity = ent.text.strip().lower()
            if normalized_entity not in seen_entities:
                seen_entities.add(normalized_entity)
                entities.append({
                    'PUBMED_ID': pubmed_id,
                    'TITLE': title,
                    'ABSTRACT': abstract,
                    'PUBLICATION DATE': year,
                    'ENTITY': normalized_entity,
                    'LABEL': ent.label_
                })
    except Exception as e:
        print(f"[ERROR] Error extracting entities for PUBMED_ID {pubmed_id}: {e}")
    return entities
