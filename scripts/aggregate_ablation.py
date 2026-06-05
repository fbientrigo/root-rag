import json
import glob
from pathlib import Path

def main():
    configs = ["A_baseline_current", "B_baseline_harness_only", "C_multi_chunk_concat", "D_multi_chunk_rrf_no_dedup", "E_forest_current", "F_forest_enhanced"]
    
    results = {}
    diagnostics = []
    
    for c in configs:
        path = f"reports/retrieval_forest/eval_{c}.json"
        if not Path(path).exists():
            continue
        with open(path) as f:
            data = json.load(f)
            results[c] = data["summary"]
            for q in data["per_query"]:
                q["config"] = c
                diagnostics.append(q)
                
    # Save results
    with open("artifacts/retrieval_forest/ablation_attribution_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Save diagnostics
    with open("artifacts/retrieval_forest/ablation_per_query_diagnostics.jsonl", "w") as f:
        for d in diagnostics:
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    main()
