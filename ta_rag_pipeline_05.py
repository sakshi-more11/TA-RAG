import faiss
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------

INDEX_PATH = "vector_store/vector_store.index"
METADATA_PATH = "vector_store/metadata.json"

EMBEDDING_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "llama3.2:1b"

TOP_K_STAGE1 = 8
TOP_K_STAGE2 = 5

CONFIDENCE_THRESHOLD = 0.65
CRITICAL_CLAIM_THRESHOLD = 0.5
MIN_EVIDENCE_COVERAGE = 0.60

# ---------------- LOAD MODELS ----------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ============================================================
# 1️⃣ MULTI-STAGE RETRIEVAL
# ============================================================

def semantic_retrieve(query, top_k=TOP_K_STAGE1):
    query_embedding = embedding_model.encode([f"query: {query}"])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx]["text"])

    return results


def rerank(query, chunks, top_k=TOP_K_STAGE2):
    # Lightweight cosine reranking
    query_emb = embedding_model.encode([f"query: {query}"])
    chunk_embs = embedding_model.encode(
        [f"passage: {c}" for c in chunks]
    )

    sims = cosine_similarity(query_emb, chunk_embs)[0]
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)

    return [c[0] for c in ranked[:top_k]]


# ============================================================
# 2️⃣ GENERATION
# ============================================================

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a regulatory compliance assistant.
Answer strictly using the provided context.
If information is not available, say:
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
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
    )
    if response.status_code != 200: #handling llm error due to h/w constraints
        print("HTTP Error:", response.text)
        return "Generation failed."

    data = response.json()
    # return response.json()["response"].strip()
    if "response" in data:
        return data["response"].strip()
    else:
        print("Ollama error during generation:", data)
        return "Generation failed."


# ============================================================
# 3️⃣ CLAIM DECOMPOSITION
# ============================================================

def decompose_claims(answer):
    sentences = answer.split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ============================================================
# 4️⃣ EVIDENCE VERIFICATION
# ============================================================

def compute_support_score(claim, retrieved_chunks):

    claim_embedding = embedding_model.encode([f"query: {claim}"])
    chunk_embeddings = embedding_model.encode(
        [f"passage: {c}" for c in retrieved_chunks]
    )

    similarities = cosine_similarity(claim_embedding, chunk_embeddings)[0]

    supported = [sim for sim in similarities if sim > 0.60]

    # Formula from paper:
    # S_i = sum(evidence support) / total retrieved chunks

    S_i = len(supported) / len(retrieved_chunks)

    return S_i, similarities


# ============================================================
# 5️⃣ CONFIDENCE SCORING
# ============================================================

def compute_confidence(support_scores):
    # Confidence = (1/n) * Σ S_i
    if len(support_scores) == 0:
        return 0
    return sum(support_scores) / len(support_scores)


# ============================================================
# 6️⃣ CALIBRATED REJECTION
# ============================================================

def apply_rejection_policy(confidence, support_scores, evidence_coverage):

    if confidence < CONFIDENCE_THRESHOLD:
        return True

    if any(s < CRITICAL_CLAIM_THRESHOLD for s in support_scores):
        return True

    if evidence_coverage < MIN_EVIDENCE_COVERAGE:
        return True

    return False


# ============================================================
# 7️⃣ FULL TA-RAG PIPELINE
# ============================================================

def ta_rag(query):

    stage1_chunks = semantic_retrieve(query)
    stage2_chunks = rerank(query, stage1_chunks)

    answer = generate_answer(query, stage2_chunks)

    claims = decompose_claims(answer)

    support_scores = []
    coverage_counter = 0

    for claim in claims:
        S_i, similarities = compute_support_score(claim, stage2_chunks)
        support_scores.append(S_i)

        if S_i > 0:
            coverage_counter += 1

    confidence = compute_confidence(support_scores)

    evidence_coverage = coverage_counter / len(claims) if claims else 0

    reject = apply_rejection_policy(
        confidence,
        support_scores,
        evidence_coverage
    )

    if reject:
        final_output = "Insufficient verified evidence available in current regulatory corpus."
    else:
        final_output = answer

    return {
        "query": query,
        "answer": answer,
        "final_output": final_output,
        "claims": claims,
        "support_scores": support_scores,
        "confidence": confidence,
        "evidence_coverage": evidence_coverage,
        "rejected": reject
    }


# ---------------- MAIN ----------------

if __name__ == "__main__":
    q = input("Enter query: ")
    result = ta_rag(q)

    print("\nFinal Output:\n", result["final_output"])
    print("\nConfidence:", result["confidence"])
    print("Rejected:", result["rejected"])
