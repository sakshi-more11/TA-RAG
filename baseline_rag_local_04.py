import faiss
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
index_path = "vector_store/vector_store.index"
metadata_path = "vector_store/metadata.json"
embedding_model = "intfloat/e5-base-v2"  # Lightweight, good for regulatory text
OLLAMA_MODEL = "llama3.2:1b"
#we may chnage llama3.2:1b to ollama pull phi3:mini, not able to use mistral 7B due to h/w constrints
TOP_K = 3

# ---------------- LOAD ----------------
model = SentenceTransformer(embedding_model)
index = faiss.read_index(index_path)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ---------------- RETRIEVE ----------------
def retrieve(query, top_k=TOP_K):
    query_embedding = model.encode([f"query: {query}"])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    retrieved_chunks = []
    for idx in I[0]:
        retrieved_chunks.append(metadata[idx]["text"])

    return retrieved_chunks


# ---------------- GENERATE (Ollama) ----------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a regulatory compliance assistant.
Answer ONLY using the provided context.
If answer is not found, say:
Information not found in provided documents.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    if response.status_code != 200:
        print("Ollama API Error:", response.text)
        return "Generation failed."

    data = response.json()

    # Safely extract response text
    answer_text = data.get("response", None)

    if answer_text is None:
        print("Unexpected API output:", data)
        return "Generation failed."

    return answer_text.strip()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    query = input("Enter your query: ")

    chunks = retrieve(query)
    answer = generate_answer(query, chunks)

    print("\nRetrieved Context:\n")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...")

    print("\nGenerated Answer:\n")
    print(answer)
