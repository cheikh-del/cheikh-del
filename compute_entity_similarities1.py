import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import numpy as np
import time


# Charger BioBERT
def load_biobert():
    """Charge le modèle BioBERT."""
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] BioBERT loaded on {device}")
    return tokenizer, model, device


# Prétraitement des embeddings optimisé
def preprocess_embeddings_optimized(embeddings_df):
    """
    Prétraite les embeddings pour garantir des dimensions uniformes et rapides.
    """
    embeddings_df = embeddings_df.select_dtypes(include=[float, int])

    if embeddings_df.empty:
        raise ValueError("[ERROR] No numeric data found in the embeddings file.")

    embeddings_matrix = embeddings_df.to_numpy(dtype=np.float32)  # Forcer en float32
    if embeddings_matrix.ndim != 2:
        raise ValueError("[ERROR] Embeddings must be a 2D matrix.")

    return pd.DataFrame(embeddings_matrix, index=embeddings_df.index)


# Calcul des similarités pour un lot
def process_batch(batch, embeddings_cache):
    """Calcule les similarités pour un lot de données."""
    results = []
    for _, row in batch.iterrows():
        source_embedding = embeddings_cache.get(row["SOURCE"])
        target_embedding = embeddings_cache.get(row["TARGET"])

        if source_embedding is not None and target_embedding is not None and len(source_embedding) == len(target_embedding):
            # Convertir explicitement en float32 pour éviter les erreurs de type
            source_embedding = torch.tensor(np.array(source_embedding, dtype=np.float32)).unsqueeze(0)
            target_embedding = torch.tensor(np.array(target_embedding, dtype=np.float32)).unsqueeze(0)

            similarity = cosine_similarity(source_embedding, target_embedding).item()
            results.append({
                "PUBMED_ID": row["PUBMED_ID"],
                "SOURCE": row["SOURCE"],
                "TARGET": row["TARGET"],
                "SOURCE_TYPE": row["SOURCE_TYPE"],
                "TARGET_TYPE": row["TARGET_TYPE"],
                "COOCCURRENCE": row["COOCCURRENCE"],
                "SOURCE_OCCURRENCE": row.get("SOURCE_OCCURRENCE", 0),
                "TARGET_OCCURRENCE": row.get("TARGET_OCCURRENCE", 0),
                "SIMILARITY": similarity
            })
    return results


# Calcul des similarités
def compute_entity_similarities(input_file, embeddings_matrix_file, output_directory, batch_size=2000, n_jobs=-1):
    """Calcule les similarités entre les entités en utilisant leurs embeddings."""
    try:
        start_time = time.time()

        # Charger les embeddings globaux
        print(f"[INFO] Loading global embeddings from {embeddings_matrix_file}")
        global_embeddings_df = pd.read_csv(embeddings_matrix_file, index_col=0)

        print("[INFO] Preprocessing embeddings...")
        global_embeddings_df = preprocess_embeddings_optimized(global_embeddings_df)
        global_embeddings = {entity: global_embeddings_df.loc[entity].values for entity in global_embeddings_df.index}

        # Charger les données
        print(f"[INFO] Loading input data from {input_file}")
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip().str.upper()

        # Vérification de la présence des colonnes nécessaires
        required_columns = ["PUBMED_ID", "SOURCE", "TARGET", "SOURCE_TYPE", "TARGET_TYPE", "COOCCURRENCE"]
        for col in ["SOURCE_OCCURRENCE", "TARGET_OCCURRENCE"]:
            if col not in df.columns:
                df[col] = 0  # Ajouter avec valeur par défaut
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"[ERROR] Missing required columns: {missing_columns}")

        # Diviser en lots
        batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
        print(f"[INFO] Processing {len(batches)} batches of size {batch_size}")

        # Calculer les similarités en parallèle
        similarity_data = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch, global_embeddings) for batch in tqdm(batches, desc="[INFO] Processing batches")
        )
        similarity_data = [item for sublist in similarity_data for item in sublist]

        output_file = os.path.join(output_directory, "entity_similarities.csv")
        os.makedirs(output_directory, exist_ok=True)
        pd.DataFrame(similarity_data).to_csv(output_file, index=False)

        elapsed_time = time.time() - start_time
        print(f"[SUCCESS] Entity similarities saved to: {output_file}")
        print(f"[INFO] Processing completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        print(f"[ERROR] Error computing similarities: {e}")


# Exemple d'appel
if __name__ == "__main__":
    input_file = "path_to_input_file.csv"
    embeddings_matrix_file = "path_to_embeddings_matrix.csv"
    output_directory = "path_to_output_directory"
    compute_entity_similarities(input_file, embeddings_matrix_file, output_directory)
