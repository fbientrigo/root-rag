import subprocess
import json

queries = {
    "q04": [
        "make_nTuple",
        "make_nTuple.py",
        "make_nTuple_SBT",
        "front side",
        "cavern",
        "muonDIS front",
        "muonDIS side",
        "muonDIS cavern"
    ],
    "q05": [
        "InactivateMuonProcesses",
        "muIoni",
        "SetProcessActivation",
        "muon process activation",
        "muBrems",
        "muPairProd",
        "muon ionisation"
    ],
    "q06": [
        "SBT",
        "UBT",
        "DOCA",
        "IP10",
        "IP250",
        "impact parameter",
        "veto",
        "nDoF chi2 ptrack",
        "fiducial wall DOCA"
    ],
    "q07": [
        "TGeoManager",
        "AddNode",
        "GetVolume",
        "FindNode",
        "ShipGeo",
        "FairShip geometry",
        "material budget",
        "decay vessel",
        "cave wall",
        "fiducial volume"
    ]
}

results = {}

for qid, qlist in queries.items():
    results[qid] = {}
    for q in qlist:
        cmd = [
            "root-rag", "ask", q,
            "--top-k", "3",
            "--index-dir", "data/indexes_fairship",
            "--root-ref", "master"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                results[qid][q] = {
                    "hits": res.stdout,
                    "exit_code": 0
                }
            else:
                results[qid][q] = {
                    "hits": None,
                    "exit_code": res.returncode
                }
        except Exception as e:
            results[qid][q] = {
                "hits": str(e),
                "exit_code": -1
            }

with open("atomic_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done.")
