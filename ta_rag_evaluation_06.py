from ta_rag_pipeline_05 import ta_rag
import json
from datetime import datetime

# ------------------------------------------------
# 30 Queries
# ------------------------------------------------

test_queries = [

    # -------- 15 In-Scope --------
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

    # -------- 5 Multi-Hop --------
    {"query": "How does the Lead Bank Scheme ensure coordination between banks and government agencies?", "type": "multi-hop"},
    {"query": "What responsibilities do Directors have regarding RBI inspection reports?", "type": "multi-hop"},
    {"query": "How are DCC and DLRC meetings structured differently?", "type": "multi-hop"},
    {"query": "How does SLBC improve financial inclusion?", "type": "multi-hop"},
    {"query": "What mechanisms exist to improve credit absorption capacity?", "type": "multi-hop"},

    # -------- 5 Ambiguous --------
    {"query": "What are the responsibilities of banks under RBI?", "type": "ambiguous"},
    {"query": "How does RBI monitor rural banking?", "type": "ambiguous"},
    {"query": "What are governance requirements for banks?", "type": "ambiguous"},
    {"query": "How does RBI ensure financial discipline?", "type": "ambiguous"},
    {"query": "What is the role of committees in banking supervision?", "type": "ambiguous"},

    # -------- 5 Out-of-Scope --------
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

correct_rejections = 0
total_out_scope = 0

results = []

print("Running batch evaluation...\n")

for i, item in enumerate(test_queries):

    print(f"Processing {i+1}/30")

    result = ta_rag(item["query"])
    result["type"] = item["type"]

    results.append(result)

    claims = result["claims"]
    scores = result["support_scores"]

    for s in scores:
        total_claims += 1
        if s == 0:
            unsupported_claims += 1
        else:
            supported_claims += 1

    if item["type"] == "out-of-scope":
        total_out_scope += 1
        if result["rejected"]:
            correct_rejections += 1

# ------------------------------------------------
# Compute Metrics
# ------------------------------------------------

hallucination_rate = unsupported_claims / total_claims if total_claims else 0
precision = supported_claims / total_claims if total_claims else 0
rejection_accuracy = correct_rejections / total_out_scope if total_out_scope else 0

print("\n--- FINAL METRICS ---")
print("Hallucination Rate:", round(hallucination_rate, 4))
print("Precision:", round(precision, 4))
print("Rejection Accuracy:", round(rejection_accuracy, 4))

# ------------------------------------------------
# Save Results
# ------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"evaluation_results_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to {output_file}")
