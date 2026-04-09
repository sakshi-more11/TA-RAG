from baseline_rag_local_04 import retrieve, generate_answer
from ta_rag_pipeline_05 import decompose_claims, compute_support_score
import json
from datetime import datetime

# ------------------------------------------------
# Same 30 Queries
# ------------------------------------------------

test_queries = [
    {"query": "What is the objective of the Lead Bank Scheme introduced by RBI?", "type": "in-scope"},
    {"query": "Who chairs the District Consultative Committee meetings?", "type": "in-scope"},
    {"query": "How frequently are BLBC meetings conducted?", "type": "in-scope"},
    {"query": "What is the minimum professional director requirement for UCBs?", "type": "in-scope"},
    {"query": "Which Act empowers RBI to regulate UCBs?", "type": "in-scope"},
    {"query": "What is the role of the Lead District Manager?", "type": "in-scope"},
    {"query": "Who chairs SLBC meetings?", "type": "in-scope"},
    {"query": "What is the purpose of Audit Committee of the Board?", "type": "in-scope"},
    {"query": "What are the eligibility restrictions for becoming a director of a UCB?", "type": "in-scope"},
    {"query": "What is RBI’s instruction regarding honorary titles at Board level?", "type": "in-scope"},
    {"query": "What is the purpose of District Level Review Committee?", "type": "in-scope"},
    {"query": "What is the frequency of SLBC meetings?", "type": "in-scope"},
    {"query": "What is the role of private sector banks in Lead Bank Scheme?", "type": "in-scope"},
    {"query": "What is the purpose of Financial Literacy Centres?", "type": "in-scope"},
    {"query": "What is the requirement regarding Board of Management in UCBs?", "type": "in-scope"},

    {"query": "How does the Lead Bank Scheme ensure coordination between banks and government agencies?", "type": "multi-hop"},
    {"query": "What responsibilities do Directors have regarding RBI inspection reports?", "type": "multi-hop"},
    {"query": "How are DCC and DLRC meetings structured differently?", "type": "multi-hop"},
    {"query": "How does SLBC improve financial inclusion?", "type": "multi-hop"},
    {"query": "What mechanisms exist to improve credit absorption capacity?", "type": "multi-hop"},

    {"query": "What are the responsibilities of banks under RBI?", "type": "ambiguous"},
    {"query": "How does RBI monitor rural banking?", "type": "ambiguous"},
    {"query": "What are governance requirements for banks?", "type": "ambiguous"},
    {"query": "How does RBI ensure financial discipline?", "type": "ambiguous"},
    {"query": "What is the role of committees in banking supervision?", "type": "ambiguous"},

    {"query": "What are RBI guidelines on cryptocurrency regulation?", "type": "out-of-scope"},
    {"query": "What is capital adequacy ratio under Basel III?", "type": "out-of-scope"},
    {"query": "What are RBI digital lending guidelines?", "type": "out-of-scope"},
    {"query": "What are penalties for KYC non-compliance?", "type": "out-of-scope"},
    {"query": "What is RBI monetary policy stance for 2025?", "type": "out-of-scope"},
]

# ------------------------------------------------
# Metrics Counters
# ------------------------------------------------

total_claims = 0
unsupported_claims = 0
supported_claims = 0

total_answers = 0
hallucinated_answers = 0

results = []

print("Running Baseline Verified Evaluation...\n")

for i, item in enumerate(test_queries):

    print(f"Processing {i+1}/30")

    # Step 1: Baseline generation
    chunks = retrieve(item["query"])
    answer = generate_answer(item["query"], chunks)

    # Step 2: Claim decomposition
    claims = decompose_claims(answer)

    answer_has_hallucination = False
    support_scores = []

    for claim in claims:
        S_i, _ = compute_support_score(claim, chunks)
        support_scores.append(S_i)

        total_claims += 1

        if S_i == 0:
            unsupported_claims += 1
            answer_has_hallucination = True
        else:
            supported_claims += 1

    total_answers += 1

    if answer_has_hallucination:
        hallucinated_answers += 1

    results.append({
        "query": item["query"],
        "type": item["type"],
        "answer": answer,
        "claims": claims,
        "support_scores": support_scores
    })

# ------------------------------------------------
# Compute Metrics
# ------------------------------------------------

claim_level_hr = unsupported_claims / total_claims if total_claims else 0
precision = supported_claims / total_claims if total_claims else 0
binary_hr = hallucinated_answers / total_answers if total_answers else 0

print("\n--- BASELINE METRICS ---")
print("Claim-Level Hallucination Rate:", round(claim_level_hr, 4))
print("Precision:", round(precision, 4))
print("Binary Hallucination Rate:", round(binary_hr, 4))

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"baseline_verified_results_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to {output_file}")
