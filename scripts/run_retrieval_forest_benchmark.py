#!/usr/bin/env python3
"""Run retrieval forest benchmark and compare configurations."""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

QUERIES_FILE = "configs/benchmark_queries_fairship_muondis_retrieval_forest.json"
QRELS_FILE = "configs/benchmark_qrels_fairship_muondis_retrieval_forest.jsonl"
OUTPUT_ROOT = Path("evidence/retrieval_forest_benchmark")

CONFIGS = [
    {
        "name": "A_baseline_current",
        "backend": "lexical",
        "forest": None,
        "fusion": "rrf",
        "dedup": "line_overlap",
        "tie_breaker": "stable",
        "profile": "fairship",
        "root_ref": "master"
    },
    {
        "name": "B_baseline_harness_only",
        "backend": "forest",
        "forest": "baseline_current",
        "fusion": "rrf",
        "dedup": "line_overlap",
        "tie_breaker": "enhanced",
        "profile": "fairship",
        "root_ref": "master"
    },
    {
        "name": "C_multi_chunk_concat",
        "backend": "forest",
        "forest": "small_80_20,medium_180_40,large_420_80",
        "fusion": "concat",
        "dedup": "none",
        "tie_breaker": "stable",
        "profile": "fairship",
        "root_ref": "master"
    },
    {
        "name": "D_multi_chunk_rrf_no_dedup",
        "backend": "forest",
        "forest": "small_80_20,medium_180_40,large_420_80",
        "fusion": "rrf",
        "dedup": "none",
        "tie_breaker": "stable",
        "profile": "fairship",
        "root_ref": "master"
    },
    {
        "name": "E_forest_current",
        "backend": "forest",
        "forest": "small_80_20,medium_180_40,large_420_80",
        "fusion": "rrf",
        "dedup": "line_overlap",
        "tie_breaker": "stable",
        "profile": "fairship",
        "root_ref": "master"
    },
    {
        "name": "F_forest_enhanced",
        "backend": "forest",
        "forest": "small_80_20,medium_180_40,large_420_80",
        "fusion": "rrf",
        "dedup": "line_overlap",
        "tie_breaker": "enhanced",
        "profile": "fairship",
        "root_ref": "master"
    }
]

def run_query(query: str, config: Dict[str, Any], output_path: Path):
    cmd = [
        ".venv/Scripts/python.exe", "-m", "root_rag.cli", "search",
        query,
        "--profile", config["profile"],
        "--root-ref", config["root_ref"],
        "--retrieval-backend", config["backend"],
        "--fusion", config["fusion"],
        "--dedup", config["dedup"],
        "--tie-breaker", config["tie_breaker"],
        "--top-k", "10",
        "--json"
    ]
    if config["forest"]:
        cmd.extend(["--retrieval-forest", config["forest"]])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    latency = (time.time() - start_time) * 1000
    
    if result.returncode != 0:
        logger.error(f"Search failed for query '{query}' in config '{config['name']}': {result.stderr}")
        return None, latency
    
    try:
        data = json.loads(result.stdout)
        return data, latency
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON for query '{query}': {result.stdout}")
        return None, latency

def main():
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    all_results = {}

    for config in CONFIGS:
        logger.info(f"Running benchmark for config: {config['name']}")
        config_dir = OUTPUT_ROOT / config["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "config": config,
            "results": []
        }
        
        latencies = []
        for q in queries:
            output_file = config_dir / f"{q['id']}.json"
            hits, latency = run_query(q["query"], config, output_file)
            latencies.append(latency)
            
            if hits:
                with open(output_file, "w") as f:
                    json.dump(hits, f, indent=2)
                
                manifest["results"].append({
                    "query_id": q["id"],
                    "query_text": q["query"],
                    "artifact_path": str(output_file.relative_to(OUTPUT_ROOT)),
                    "latency_ms": latency
                })
        
        with open(config_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Run evaluation
        eval_output = Path(f"reports/retrieval_forest/eval_{config['name']}.json")
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        
        eval_cmd = [
            ".venv/Scripts/python.exe", "scripts/evaluate_muon_dis_retrieval.py",
            "--evidence-dir", str(config_dir),
            "--golden", QUERIES_FILE,
            "--qrels", QRELS_FILE,
            "--output", str(eval_output)
        ]
        subprocess.run(eval_cmd)
        logger.info(f"Evaluation saved to {eval_output}")

if __name__ == "__main__":
    main()
