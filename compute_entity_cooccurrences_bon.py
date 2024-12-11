import os
import pandas as pd
from itertools import combinations
from collections import Counter
import gc

def compute_entity_cooccurrences(corpus_file, output_directory):
    """
    Calcule les cooccurrences des entités à partir du fichier de corpus, 
    avec mapping des PUBMED_ID et séparation du calcul des cooccurrences globales.
    """
    try:
        print(f"[INFO] Loading corpus file: {corpus_file}")
        
        # Charger le corpus
        corpus = pd.read_csv(corpus_file)
        corpus.columns = corpus.columns.str.strip().str.upper()

        # Vérification des colonnes obligatoires
        required_columns = {"PUBMED_ID", "DOCUMENT"}
        if not required_columns.issubset(corpus.columns):
            raise ValueError(f"[ERROR] Missing required columns: {required_columns - set(corpus.columns)}")
        if corpus.empty:
            raise ValueError("[ERROR] The corpus file is empty.")

        # Calculer les occurrences globales des entités
        print("[INFO] Calculating global occurrences...")
        entity_occurrence_map = Counter()
        for _, row in corpus.iterrows():
            entities_with_types = eval(row["DOCUMENT"])
            for entity, _ in entities_with_types:
                entity_occurrence_map[entity] += 1

        # Calcul des cooccurrences globales
        print("[INFO] Calculating global cooccurrences...")
        global_cooccurrence_counter = Counter()

        for _, row in corpus.iterrows():
            try:
                entities_with_types = eval(row["DOCUMENT"])  # Convertir la chaîne en liste de tuples
                for (source, source_type), (target, target_type) in combinations(entities_with_types, 2):
                    # Assurer un ordre constant pour éviter les doublons inversés
                    if source > target:
                        source, source_type, target, target_type = target, target_type, source, source_type
                    global_cooccurrence_counter[(source, source_type, target, target_type)] += 1
            except Exception as e:
                print(f"[WARNING] Skipping invalid row: {e}")

        # Préparer les données avec PUBMED_ID
        print("[INFO] Mapping cooccurrences with PUBMED_ID...")
        cooccurrence_data = []
        for _, row in corpus.iterrows():
            try:
                pubmed_id = row["PUBMED_ID"]
                entities_with_types = eval(row["DOCUMENT"])
                for (source, source_type), (target, target_type) in combinations(entities_with_types, 2):
                    if source > target:
                        source, source_type, target, target_type = target, target_type, source, source_type
                    count = global_cooccurrence_counter[(source, source_type, target, target_type)]
                    cooccurrence_data.append({
                        "PUBMED_ID": pubmed_id,
                        "SOURCE": source,
                        "SOURCE_TYPE": source_type,
                        "TARGET": target,
                        "TARGET_TYPE": target_type,
                        "COOCCURRENCE": count,
                        "SOURCE_OCCURRENCE": entity_occurrence_map[source],
                        "TARGET_OCCURRENCE": entity_occurrence_map[target]
                    })
            except Exception as e:
                print(f"[WARNING] Skipping invalid row: {e}")

        # Transformer en DataFrame
        cooccurrence_df = pd.DataFrame(cooccurrence_data)

        # Sauvegarder les cooccurrences
        cooccurrence_file = os.path.join(output_directory, "entity_cooccurrences.csv")
        os.makedirs(output_directory, exist_ok=True)
        cooccurrence_df.to_csv(cooccurrence_file, index=False)
        print(f"[SUCCESS] Cooccurrences saved to: {cooccurrence_file}")

        return cooccurrence_file
    except Exception as e:
        print(f"[ERROR] Failed to compute cooccurrences: {e}")
        return None
