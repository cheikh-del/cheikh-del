import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import torch
from joblib import Parallel, delayed
import os
import time
from tqdm import tqdm

def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] BioBERT loaded on device: {device}")
    return tokenizer, model, device

def get_embedding(entity, tokenizer, model, device, max_length=512):
    inputs = tokenizer(entity, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def process_batch(batch, embeddings_cache, source_occurrences, target_occurrences):
    results = []
    for _, row in batch.iterrows():
        source_embedding = embeddings_cache[row["SOURCE"]]
        target_embedding = embeddings_cache[row["TARGET"]]
        similarity = cosine_similarity(
            torch.tensor(source_embedding).unsqueeze(0),
            torch.tensor(target_embedding).unsqueeze(0)
        ).item()
        results.append({
            "PUBMED_ID": row["PUBMED_ID"],
            "SOURCE": row["SOURCE"],
            "TARGET": row["TARGET"],
            "SOURCE_TYPE": row["SOURCE_TYPE"],
            "TARGET_TYPE": row["TARGET_TYPE"],
            "COOCCURRENCE": row["COOCCURRENCE"],
            "SIMILARITY": similarity,
            "SOURCE_OCCURRENCE": source_occurrences[row["SOURCE"]],
            "TARGET_OCCURRENCE": target_occurrences[row["TARGET"]]
        })
    return results

def compute_entity_similarities(input_file, output_directory, batch_size=1000, n_jobs=-1):
    try:
        tokenizer, model, device = load_biobert()
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip().str.upper()
        source_occurrences = df["SOURCE"].value_counts().to_dict()
        target_occurrences = df["TARGET"].value_counts().to_dict()
        unique_entities = set(df["SOURCE"]).union(set(df["TARGET"]))

        embeddings_cache = {
            entity: get_embedding(entity, tokenizer, model, device)
            for entity in tqdm(unique_entities, desc="[INFO] Precomputing embeddings")
        }

        batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
        similarity_data = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch, embeddings_cache, source_occurrences, target_occurrences)
            for batch in tqdm(batches, desc="[INFO] Processing batches")
        )

        similarity_data = [item for sublist in similarity_data for item in sublist]
        output_file = os.path.join(output_directory, "entity_similarities.csv")
        os.makedirs(output_directory, exist_ok=True)
        pd.DataFrame(similarity_data).to_csv(output_file, index=False)
        print(f"[SUCCESS] Entity similarities saved to {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to compute entity similarities: {e}")
