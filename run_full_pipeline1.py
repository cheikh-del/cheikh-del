import os
import pandas as pd
from pubmed_pipeline.process_bionlp import process_compiled_file_with_bionlp
from pubmed_pipeline.compute_entity_cooccurrences import compute_entity_cooccurrences_with_singular
from pubmed_pipeline.compute_entity_similarities import compute_entity_similarities
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# === Configuration des chemins ===
BASE_DIR = 'C:/collecte/biolm/Test'
test_file_path = os.path.join(BASE_DIR, "Input")
output_directory = os.path.join(BASE_DIR, "Output")
entities_directory = os.path.join(output_directory, "entities")
embeddings_directory = os.path.join(output_directory, "embeddings")
similarities_directory = os.path.join(output_directory, "similarities")
merged_entities_file = os.path.join(entities_directory, "pubmed_cosmetics_entities_all.csv")
cooccurrences_file = os.path.join(embeddings_directory, "entity_cooccurrences.csv")
embeddings_file = os.path.join(embeddings_directory, "entity_embeddings.csv")
similarities_file = os.path.join(similarities_directory, "entity_similarities_with_cooccurrences.csv")

# === Utilitaire : Création de répertoires ===
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Directory created: {directory}")
    else:
        print(f"[INFO] Directory already exists: {directory}")

# === Fusion des entités ===
def merge_entities():
    """Fusionne les fichiers d'entités extraites."""
    if not os.path.exists(merged_entities_file):
        print("[INFO] Merging entity files...")
        entity_files = [os.path.join(entities_directory, f) for f in os.listdir(entities_directory) if f.endswith('_entities.csv')]
        if entity_files:
            all_entities = pd.concat([pd.read_csv(f) for f in entity_files], ignore_index=True)
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
        print("[INFO] Generating cooccurrences...")
        compute_entity_cooccurrences_with_singular(merged_entities_file, embeddings_directory)
        print(f"[SUCCESS] Cooccurrences saved to {cooccurrences_file}")
    else:
        print("[INFO] Cooccurrences already generated.")

# === Génération des embeddings ===
def generate_entity_embeddings():
    """Génère les embeddings pour les entités uniques."""
    ensure_directory_exists(embeddings_directory)
    if not os.path.exists(embeddings_file):
        print("[INFO] Generating entity embeddings...")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to("cuda" if torch.cuda.is_available() else "cpu")
        
        df = pd.read_csv(merged_entities_file)
        unique_entities = df["ENTITY"].dropna().unique()

        embeddings = {}
        for entity in tqdm(unique_entities, desc="[INFO] Generating embeddings"):
            inputs = tokenizer(entity, return_tensors="pt", truncation=True, padding=True).to(model.device)
            outputs = model(**inputs)
            embeddings[entity] = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

        pd.DataFrame.from_dict(embeddings, orient="index").to_csv(embeddings_file)
        print(f"[SUCCESS] Entity embeddings saved to {embeddings_file}")
    else:
        print("[INFO] Entity embeddings already generated.")

# === Calcul des similarités ===
def compute_similarities():
    """Calcule les similarités entre les entités."""
    ensure_directory_exists(similarities_directory)
    if not os.path.exists(similarities_file):
        print("[INFO] Calculating entity similarities...")
        try:
            compute_entity_similarities(cooccurrences_file, embeddings_file, similarities_directory)
            print(f"[SUCCESS] Similarities saved to {similarities_file}")
        except Exception as e:
            print(f"[ERROR] Failed to compute similarities: {e}")
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
    generate_entity_embeddings()
    compute_similarities()

# Lancer le pipeline
run_pipeline()
