import os
import pandas as pd
from datetime import datetime, timedelta
from pubmed_pipeline.process_bionlp import process_compiled_file_with_bionlp
from pubmed_pipeline.compute_entity_cooccurrences import compute_entity_cooccurrences_with_singular
from pubmed_pipeline.compute_entity_similarities import compute_entity_similarities

# === Configuration des chemins ===
BASE_DIR = 'C:/collecte/biolm/Test'
test_file_path = os.path.join(BASE_DIR, "Input")
output_directory = os.path.join(BASE_DIR, "Output")
entities_directory = os.path.join(output_directory, "entities")
embeddings_directory = os.path.join(output_directory, "embeddings")
similarities_directory = os.path.join(output_directory, "similarities")
merged_entities_file = os.path.join(entities_directory, "pubmed_cosmetics_entities_all.csv")
cooccurrences_file = os.path.join(embeddings_directory, "entity_cooccurrences.csv")
similarities_file = os.path.join(similarities_directory, "entity_similarities_with_cooccurrences.csv")

# === Utilitaire : Création de répertoires ===
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# === Fusion des entités ===
def merge_entities():
    """Fusionne les fichiers d'entités extraites."""
    if not os.path.exists(merged_entities_file):
        entity_files = [os.path.join(entities_directory, f) for f in os.listdir(entities_directory) if f.endswith('_entities.csv')]
        if entity_files:
            all_entities = pd.concat([pd.read_csv(f) for f in entity_files], ignore_index=True)
            all_entities["ENTITY"] = all_entities["ENTITY"].astype(str)
            all_entities["LABEL"] = all_entities["LABEL"].astype(str)
            all_entities.to_csv(merged_entities_file, index=False)
            print(f"[SUCCESS] Merged entities saved to {merged_entities_file}")
        else:
            print("[ERROR] No entity files to merge.")
    else:
        print("[INFO] Entity files already merged.")

# === Génération des cooccurrences ===
def generate_cooccurrences():
    """Génère les cooccurrences des entités."""
    ensure_directory_exists(embeddings_directory)
    if not os.path.exists(cooccurrences_file):
        compute_entity_cooccurrences_with_singular(merged_entities_file, embeddings_directory)
    else:
        print("[INFO] Cooccurrences already generated.")

# === Calcul des similarités ===
def compute_similarities():
    """Calcule les similarités entre les entités."""
    ensure_directory_exists(similarities_directory)
    if not os.path.exists(similarities_file):
        compute_entity_similarities(cooccurrences_file, os.path.join(embeddings_directory, "entity_embeddings.csv"), similarities_directory)
    else:
        print("[INFO] Similarities already computed.")

# === Pipeline principal ===
def run_pipeline():
    """Exécute le pipeline complet."""
    ensure_directory_exists(test_file_path)
    ensure_directory_exists(output_directory)

    # Étapes principales
    merge_entities()
    generate_cooccurrences()
    compute_similarities()

# Lancer le pipeline
run_pipeline()