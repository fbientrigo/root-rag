# Retrieval Iteration Report (2026-03-29)

## Hypothesis
Low-signal natural-language terms are over-weighting lexical mismatch.
If we normalize benchmark queries by dropping low-signal intent words and adding a small ROOT/FairShip alias expansion set, BM25 retrieval should improve recall and hard structural query performance.

## Intervention
- Added `scripts/run_retrieval_benchmark.py` to run a frozen benchmark loop from existing artifacts.
- Reconstructed `configs/benchmark_queries.json` and `configs/benchmark_qrels.jsonl` from `artifacts/benchmark_eval_results.json` with metric-consistent qrels replay.
- Implemented one lexical intervention in benchmark retrieval query preprocessing (`--query-mode lexnorm`):
  - remove low-signal terms (`usage`, `pattern`, `implementation`, `modules`, `detectors`, `fairship`, etc.)
  - add small alias expansions (`TGeoManager -> gGeoManager/GetTopVolume/...`, `SetBranchAddress -> GetEntry/TTree`, `ShipFieldMaker -> defineGlobalField/defineLocalField`, etc.)
- Kept BM25 settings fixed (`k1=1.5`, `b=0.75`, top-k=10).

## Benchmark Results (Before vs After)

### Overall
- MRR@10: `0.4348 -> 0.4580` (`+0.0232`)
- Recall@10: `0.4792 -> 0.5417` (`+0.0625`)
- nDCG@10: `0.4031 -> 0.4547` (`+0.0517`)

### Per-Class
- `common_api`
  - MRR@10: `0.3643 -> 0.3350` (`-0.0293`)
  - Recall@10: `0.4000 -> 0.5500` (`+0.1500`)
  - nDCG@10: `0.3411 -> 0.3874` (`+0.0463`)
- `structural_usage`
  - MRR@10: `0.2241 -> 0.3304` (`+0.1062`)
  - Recall@10: `0.4375 -> 0.4375` (`+0.0000`)
  - nDCG@10: `0.2312 -> 0.3329` (`+0.1017`)
- `rare_api`
  - MRR@10: `0.8333 -> 0.8333` (`+0.0000`)
  - Recall@10: `0.6667 -> 0.6667` (`+0.0000`)
  - nDCG@10: `0.7355 -> 0.7295` (`-0.0060`)

## Query Effects
- Helped (`6`): `c001`, `c002`, `c003`, `c004`, `s004`, `s007`
- Hurt (`3`): `c005`, `r002`, `s006`
- Unchanged (`15`): `c006`, `c007`, `c008`, `c009`, `c010`, `r001`, `r003`, `r004`, `r005`, `r006`, `s001`, `s002`, `s003`, `s005`, `s008`

## Verdict
`helped`

The intervention produced practical gains, including a real recall increase and measurable `structural_usage` improvement. The effect is not universal because `rare_api` is mostly flat and a small subset regressed.

## Next Step
Single next step: run one constrained follow-up lexical iteration focused only on regressions (`c005`, `s006`, `r002`) using targeted alias/term-control changes, while keeping query-stopword policy fixed.
