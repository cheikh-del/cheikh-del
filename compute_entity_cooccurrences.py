import os
import pandas as pd
from itertools import combinations
from collections import Counter
import gc
import inflect

# Initialiser l'objet inflect pour convertir les mots au singulier
inflect_engine = inflect.engine()

def singularize_entity(entity):
    """Transforme une entité au singulier."""
    return inflect_engine.singular_noun(entity) if inflect_engine.singular_noun(entity) else entity

def compute_entity_cooccurrences_with_singular(input_file, output_directory):
    """Calcule les cooccurrences enrichies en transformant les entités au singulier."""
    try:
        print(f"[INFO] Loading file: {input_file}")
        
        # Charger les données
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip().str.upper()

        # Vérification des colonnes requises
        required_columns = {"ENTITY", "LABEL", "PUBMED_ID"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"[ERROR] Missing required columns: {missing_columns}")
        if df.empty:
            raise ValueError("[ERROR] The input file is empty or invalid.")

        # Nettoyage des données
        print("[INFO] Cleaning and preparing data...")
        df = df.dropna(subset=required_columns).drop_duplicates()
        df["ENTITY"] = df["ENTITY"].apply(singularize_entity)

        # Compter les occurrences des entités par PUBMED_ID
        print("[INFO] Counting entity occurrences...")
        entity_occurrences = df.groupby(["PUBMED_ID", "ENTITY"]).size().reset_index(name="OCCURRENCE")

        # Créer un corpus groupé par PUBMED_ID
        print("[INFO] Grouping entities by article...")
        corpus = df.groupby("PUBMED_ID")[["ENTITY", "LABEL"]].apply(
            lambda x: list(zip(x["ENTITY"], x["LABEL"]))
        ).reset_index(name="DOCUMENT")

        # Calcul des cooccurrences
        print("[INFO] Calculating cooccurrences...")
        cooccurrence_counter = Counter()
        for _, row in corpus.iterrows():
            entities_with_types = row["DOCUMENT"]
            unique_pairs = set()
            for pair in combinations(entities_with_types, 2):
                source, source_type = pair[0]
                target, target_type = pair[1]
                # Assurer un ordre constant pour éviter les doublons
                if source > target:
                    source, source_type, target, target_type = target, target_type, source, source_type
                unique_pairs.add((source, source_type, target, target_type, row["PUBMED_ID"]))
            # Ajouter les cooccurrences uniques
            for unique_pair in unique_pairs:
                cooccurrence_counter[unique_pair] += 1

        # Libérer la mémoire
        del corpus, df
        gc.collect()

        # Sauvegarder les cooccurrences
        print("[INFO] Saving cooccurrences...")
        cooccurrence_df = pd.DataFrame([
            {"PUBMED_ID": pubmed_id, "SOURCE": source, "SOURCE_TYPE": source_type, 
             "TARGET": target, "TARGET_TYPE": target_type, "COOCCURRENCE": count}
            for (source, source_type, target, target_type, pubmed_id), count in cooccurrence_counter.items()
        ])

        # Réorganiser les colonnes pour placer PUBMED_ID en premier
        columns_order = ["PUBMED_ID", "SOURCE", "SOURCE_TYPE", "TARGET", "TARGET_TYPE", "COOCCURRENCE"]
        cooccurrence_df = cooccurrence_df[columns_order]

        output_file = os.path.join(output_directory, "entity_cooccurrences.csv")
        cooccurrence_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Entity cooccurrences saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"[ERROR] Failed to compute entity cooccurrences: {e}")
        return None
