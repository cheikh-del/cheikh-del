import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def compute_entity_similarities(cooccurrence_file, output_directory):
    """
    Calcule les similarités entre les entités basées sur les cooccurrences,
    en incluant SOURCE_OCCURRENCE et TARGET_OCCURRENCE.
    """
    try:
        print("[INFO] Loading cooccurrence file...")
        df = pd.read_csv(cooccurrence_file)
        df.columns = df.columns.str.strip().str.upper()

        required_columns = {"PUBMED_ID", "SOURCE", "TARGET", "SOURCE_TYPE", "TARGET_TYPE", "COOCCURRENCE", "SOURCE_OCCURRENCE", "TARGET_OCCURRENCE"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

        # Charger le modèle BioBERT
        print("[INFO] Loading BioBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to("cuda" if torch.cuda.is_available() else "cpu")

        # Générer les embeddings pour toutes les entités
        print("[INFO] Generating embeddings...")
        unique_entities = pd.concat([df["SOURCE"], df["TARGET"]]).unique()
        embeddings = {}
        for entity in tqdm(unique_entities, desc="Generating embeddings"):
            inputs = tokenizer(entity, return_tensors="pt", truncation=True, padding=True).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Utiliser la moyenne sur la dernière couche cachée pour obtenir un vecteur 1D
            embeddings[entity] = outputs.last_hidden_state.mean(dim=1).cpu()

        # Calculer les similarités
        print("[INFO] Calculating similarities...")
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating similarities"):
            try:
                source_embedding = embeddings[row["SOURCE"]]
                target_embedding = embeddings[row["TARGET"]]
                # Calculer la similarité cosinus
                similarity = cosine_similarity(source_embedding, target_embedding).item()
                results.append({
                    "PUBMED_ID": row["PUBMED_ID"],
                    "SOURCE": row["SOURCE"],
                    "TARGET": row["TARGET"],
                    "SOURCE_TYPE": row["SOURCE_TYPE"],
                    "TARGET_TYPE": row["TARGET_TYPE"],
                    "COOCCURRENCE": row["COOCCURRENCE"],
                    "SOURCE_OCCURRENCE": row["SOURCE_OCCURRENCE"],
                    "TARGET_OCCURRENCE": row["TARGET_OCCURRENCE"],
                    "SIMILARITY": similarity
                })
            except KeyError as e:
                print(f"[WARNING] Missing embedding for entity: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to calculate similarity for row: {e}")

        # Sauvegarder les similarités
        output_file = os.path.join(output_directory, "entity_similarities.csv")
        os.makedirs(output_directory, exist_ok=True)
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"[SUCCESS] Similarities saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"[ERROR] Failed to compute similarities: {e}")
        return None
