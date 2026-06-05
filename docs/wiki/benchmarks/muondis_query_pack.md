# MuonDIS Query Pack Benchmark (2026-05-29)

Source report: `reports/root_rag_muondis_query_audit.md` (reports/root_rag_muondis_query_audit.md:1-62).

Query pack: `query_packs/fairship_muondis_query_pack_v1.yaml` (reports/root_rag_muondis_query_audit.md:7-8).

## Retrieval outcomes

| Query | Status | Evidence anchor |
|---|---|---|
| makeMuonDIS | **PASS** | `muonDIS/makeMuonDIS.py:130-150` (muonDIS/makeMuonDIS.py:130-150). |
| run_simScript muonDIS | **PARTIAL** | `macro/run_simScript.py:327-333` and `macro/run_simScript.py:841-861` (macro/run_simScript.py:327-333, 841-861). |
| ShipReco | **PASS** | `macro/ShipReco.py:163-175` (macro/ShipReco.py:163-175). |
| make_nTuple front side cavern | **FAIL** | No hits (reports/root_rag_muondis_query_audit.md:28-31). |
| InactivateMuonProcesses muIoni | **FAIL** | No hits (reports/root_rag_muondis_query_audit.md:28-31). |
| SBT UBT DOCA IP250 IP10 | **FAIL** | No hits (reports/root_rag_muondis_query_audit.md:28-31). |
| TGeoManager AddNode FairShip geometry | **FAIL** | No hits (reports/root_rag_muondis_query_audit.md:28-31). |

## Qrels

`docs/wiki/benchmarks/muondis_query_pack.qrels.json` (docs/wiki/benchmarks/muondis_query_pack.qrels.json:1-38).

## Notes

- Retrieval evidence does **not** upgrade runtime claims (reports/root_rag_muondis_query_audit.md:57-62).
