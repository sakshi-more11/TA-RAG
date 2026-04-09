import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
chunk_folder = "data/chunks"
index_path = "vector_store/vector_store.index"
metadata_path = "vector_store/metadata.json"
os.makedirs("vector_store", exist_ok=True)

# Load embedding model (lightweight, good for regulatory text)
model = SentenceTransformer("intfloat/e5-base-v2")

all_chunks = []
metadata = []

print("Loading chunks...")

for file in os.listdir(chunk_folder):
    if file.endswith("_chunks.json"):
        path = os.path.join(chunk_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "source_document": file.replace("_chunks.json", ".pdf"),
                "chunk_id": f"{file}_{i}",
                "text": chunk
            })

print("Generating embeddings...")

embeddings = model.encode([f"passage: {chunk}" for chunk in all_chunks]) #e5 models require prefix
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, index_path)

# Save metadata
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Vector store built successfully.")
print(f"Total vectors stored: {len(all_chunks)}")
