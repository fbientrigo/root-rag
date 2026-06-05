import json

def main():
    baseline_missing = set()
    forest_recovered = {}
    
    with open('artifacts/retrieval_forest/ablation_per_query_diagnostics.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['config'] == 'A_baseline_current' and (d['status'] == 'ERROR' or d.get('recall_at_k', 0) == 0):
                baseline_missing.add(d['query_id'])
            
            if d['config'] == 'E_forest_current' and d.get('recall_at_k', 0) > 0:
                forest_recovered[d['query_id']] = d

    print(f"Baseline missing: {baseline_missing}")
    for qid in baseline_missing:
        if qid in forest_recovered:
            d = forest_recovered[qid]
            print(f"Query {qid} ({d['query']}) recovered by forest: MRR={d.get('mrr_at_k', 0):.4f}")

if __name__ == "__main__":
    main()
