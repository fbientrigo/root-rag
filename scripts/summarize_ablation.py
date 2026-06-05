import json
import glob
from pathlib import Path

def main():
    files = sorted(glob.glob('reports/retrieval_forest/eval_[A-F]_*.json'))
    print(f"{'Config':<30} | {'Recall':<10} | {'MRR':<10} | {'Scored':<10}")
    print("-" * 65)
    
    for f in files:
        config_name = Path(f).stem.replace("eval_", "")
        with open(f) as fd:
            data = json.load(fd)
            s = data['summary']
            print(f"{config_name:<30} | {s['macro_recall_at_k']:<10.4f} | {s['macro_mrr_at_k']:<10.4f} | {s['scored_query_count']:<10}")

if __name__ == "__main__":
    main()
