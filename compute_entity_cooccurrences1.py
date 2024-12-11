import os
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict
import gc
import inflect

# Initialiser l'objet inflect pour convertir les mots au singulier
inflect_engine = inflect.engine()

def singularize_entity(entity):
    """Transforme une entité au singulier."""
    return inflect_engine.singular_noun(entity) if inflect_engine.singular_noun(entity) else entity

def compute_entity_cooccurrences_with_singular(input_file, output_directory):
    """
    Crée un corpus des entités groupées par article, transforme les entités au singulier,
    puis calcule les cooccurrences enrichies avec les types des entités.
    """
    try:
        print(f"[INFO] Loading file: {input_file}")
        
        # Charger les données en chunks pour économiser la mémoire
        df_chunks = pd.read_csv(input_file, chunksize=100000)
        all_chunks = []

        # Vérifier les colonnes et concaténer les chunks
        for chunk in df_chunks:
            chunk.columns = chunk.columns.str.strip().str.upper()
            required_columns = {"ENTITY", "LABEL", "PUBMED_ID", "CONTENT"}
            if not required_columns.issubset(chunk.columns):
                raise ValueError(f"[ERROR] Missing required columns: {required_columns - set(chunk.columns)}")
            all_chunks.append(chunk)

        # Fusionner tous les chunks
        df = pd.concat(all_chunks, ignore_index=True)
        del all_chunks  # Libérer la mémoire
        gc.collect()

        # Pré-traitement : nettoyage des données
        print("[INFO] Cleaning and preparing data...")
        df = df.dropna(subset=["ENTITY", "LABEL", "PUBMED_ID", "CONTENT"]).drop_duplicates()

        # Transformer les entités au singulier
        print("[INFO] Transforming entities to singular form...")
        df["ENTITY"] = df["ENTITY"].apply(singularize_entity)

        # Compter les occurrences des entités par PUBMED_ID
        print("[INFO] Counting occurrences of entities...")
        entity_occurrences = df.groupby(["PUBMED_ID", "ENTITY"]).size().reset_index(name="OCCURRENCE")

        # Créer un corpus groupé par PUBMED_ID
        print("[INFO] Creating corpus...")
        corpus = (
            df.groupby("PUBMED_ID")[["ENTITY", "LABEL"]]
            .apply(lambda x: list(zip(x["ENTITY"], x["LABEL"])))
            .reset_index(name="DOCUMENT")
        )

        # Sauvegarder le corpus
        corpus_file = os.path.join(output_directory, "entity_corpus_with_singular.csv")
        os.makedirs(output_directory, exist_ok=True)
        corpus.to_csv(corpus_file, index=False)
        print(f"[SUCCESS] Corpus saved to: {corpus_file}")

        # Calculer les cooccurrences
        print("[INFO] Calculating cooccurrences...")
        cooccurrence_counter = Counter()
        cooccurrence_details = defaultdict(list)  # Stocker les PUBMED_ID pour chaque paire

        for _, row in corpus.iterrows():
            pubmed_id = row["PUBMED_ID"]
            entities_with_types = row["DOCUMENT"]  # Liste de tuples (ENTITY, LABEL)
            document_pairs = set()  # Ensemble pour stocker les paires uniques dans ce document

            for (source, source_type), (target, target_type) in combinations(entities_with_types, 2):
                # Trier les entités pour éviter les doublons inversés
                if source > target:
                    source, source_type, target, target_type = target, target_type, source, source_type

                # Ajouter la paire triée au set
                document_pairs.add((source, source_type, target, target_type))

            # Mettre à jour le compteur global avec les paires uniques de ce document
            for source, source_type, target, target_type in document_pairs:
                cooccurrence_counter[(source, source_type, target, target_type)] += 1
                cooccurrence_details[(source, source_type, target, target_type)].append(pubmed_id)

        # Transformer les cooccurrences en DataFrame
        cooccurrence_results = []
        for pair, count in cooccurrence_counter.items():
            source, source_type, target, target_type = pair
            pubmed_ids = cooccurrence_details[pair]
            source_occurrence = entity_occurrences.loc[
                (entity_occurrences["ENTITY"] == source), "OCCURRENCE"
            ].sum()
            target_occurrence = entity_occurrences.loc[
                (entity_occurrences["ENTITY"] == target), "OCCURRENCE"
            ].sum()

            cooccurrence_results.append({
                "PUBMED_ID": ";".join(map(str, pubmed_ids)),  # Combiner les IDs dans une chaîne
                "SOURCE": source,
                "SOURCE_TYPE": source_type,
                "TARGET": target,
                "TARGET_TYPE": target_type,
                "COOCCURRENCE": count,
                "SOURCE_OCCURRENCE": source_occurrence,
                "TARGET_OCCURRENCE": target_occurrence,
            })

        cooccurrence_df = pd.DataFrame(cooccurrence_results)
        cooccurrence_output_file = os.path.join(output_directory, "entity_cooccurrences_with_singular.csv")
        cooccurrence_df.to_csv(cooccurrence_output_file, index=False)
        print(f"[SUCCESS] Entity cooccurrences saved to: {cooccurrence_output_file}")

        return corpus_file, cooccurrence_output_file

    except Exception as e:
        print(f"[ERROR] Failed to compute entity cooccurrences: {e}")
        return None, None
