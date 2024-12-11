import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import os
from tqdm import tqdm
import numpy as np
import time
import gc

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
    """Prétraite les embeddings pour garantir des dimensions uniformes et rapides."""
    embeddings_df = embeddings_df.select_dtypes(include=[float, int])
    if embeddings_df.empty:
        raise ValueError("[ERROR] No numeric data found in the embeddings file.")
    embeddings_matrix = embeddings_df.to_numpy(dtype=np.float32)
    if embeddings_matrix.ndim != 2:
        raise ValueError("[ERROR] Embeddings must be a 2D matrix.")
    return pd.DataFrame(embeddings_matrix, index=embeddings_df.index)

# Calcul des similarités
def compute_entity_similarities(input_file, embeddings_matrix_file, output_directory, cooccurrences_file=None, batch_size=2000):
    """Calcule les similarités entre les entités en utilisant leurs embeddings."""
    try:
        start_time = time.time()

        # Charger les embeddings globaux
        print(f"[INFO] Loading global embeddings from {embeddings_matrix_file}")
        global_embeddings_df = pd.read_csv(embeddings_matrix_file, index_col=0)
        global_embeddings_df = preprocess_embeddings_optimized(global_embeddings_df)
        global_embeddings = {entity: global_embeddings_df.loc[entity].values for entity in global_embeddings_df.index}

        # Charger les données d'entrée
        input_df = pd.read_csv(input_file if not cooccurrences_file else cooccurrences_file)
        input_df.columns = input_df.columns.str.strip().str.upper()

        # Vérifier les colonnes nécessaires
        required_columns = {"PUBMED_ID", "SOURCE", "TARGET", "SOURCE_TYPE", "TARGET_TYPE", "COOCCURRENCE"}
        missing_columns = required_columns - set(input_df.columns)
        if missing_columns:
            raise ValueError(f"[ERROR] Missing required columns: {missing_columns}")

        # Diviser en lots
        batches = [input_df.iloc[i:i + batch_size] for i in range(0, len(input_df), batch_size)]
        print(f"[INFO] Processing {len(batches)} batches of size {batch_size}")

        similarity_data = []
        for batch in tqdm(batches, desc="[INFO] Processing batches"):
            results = []
            for _, row in batch.iterrows():
                source = global_embeddings.get(row["SOURCE"])
                target = global_embeddings.get(row["TARGET"])
                if source is not None and target is not None:
                    source_tensor = torch.tensor(source, dtype=torch.float32).unsqueeze(0)
                    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
                    similarity = cosine_similarity(source_tensor, target_tensor).item()
                    results.append({
                        "PUBMED_ID": row["PUBMED_ID"],
                        "SOURCE": row["SOURCE"],
                        "SOURCE_TYPE": row["SOURCE_TYPE"],
                        "TARGET": row["TARGET"],
                        "TARGET_TYPE": row["TARGET_TYPE"],
                        "COOCCURRENCE": row["COOCCURRENCE"],
                        "SIMILARITY": similarity
                    })
                else:
                    print(f"[WARNING] Missing embeddings for SOURCE={row['SOURCE']} or TARGET={row['TARGET']}")
            similarity_data.extend(results)
            gc.collect()  # Libérer la mémoire après chaque batch

        # Sauvegarder les similarités
        output_file = os.path.join(output_directory, "entity_similarities_with_cooccurrences.csv")
        os.makedirs(output_directory, exist_ok=True)
        similarity_df = pd.DataFrame(similarity_data)
        similarity_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Similarities saved to {output_file}")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Processing completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        print(f"[ERROR] Error computing similarities: {e}")
