#!/usr/bin/env bash
set -euo pipefail

label() {
  printf '[%s] %s\n' "$1" "$2"
}

label "PREFLIGHT" "Checking FAIRSHIP and SHIPSOFT variables"
echo "FAIRSHIP=${FAIRSHIP:-UNSET}"
echo "SHIPSOFT=${SHIPSOFT:-UNSET}"

label "PREFLIGHT" "Checking FAIRSHIP path exists"
if [[ -n "${FAIRSHIP:-}" && -d "${FAIRSHIP}" ]]; then
  echo "[OK] FAIRSHIP directory exists"
else
  echo "[FAIL] FAIRSHIP is missing or not a directory"
fi

label "PREFLIGHT" "Checking SHIPSOFT path exists"
if [[ -n "${SHIPSOFT:-}" && -d "${SHIPSOFT}" ]]; then
  echo "[OK] SHIPSOFT directory exists"
else
  echo "[FAIL] SHIPSOFT is missing or not a directory"
fi

label "PREFLIGHT" "Checking required scripts"
if [[ -n "${FAIRSHIP:-}" && -f "${FAIRSHIP}/macro/run_simScript.py" ]]; then
  echo "[OK] ${FAIRSHIP}/macro/run_simScript.py"
else
  echo "[FAIL] run_simScript.py not found"
fi
if [[ -n "${FAIRSHIP:-}" && -f "${FAIRSHIP}/muonDIS/makeMuonDIS.py" ]]; then
  echo "[OK] ${FAIRSHIP}/muonDIS/makeMuonDIS.py"
else
  echo "[FAIL] makeMuonDIS.py not found"
fi

label "PREFLIGHT" "Checking MuonBack example input file"
if [[ -n "${SHIPSOFT:-}" && -f "${SHIPSOFT}/data/pythia8_Geant4_onlyMuons.root" ]]; then
  echo "[OK] ${SHIPSOFT}/data/pythia8_Geant4_onlyMuons.root"
else
  echo "[WARN] ${SHIPSOFT:-UNSET}/data/pythia8_Geant4_onlyMuons.root not found"
fi

label "PREFLIGHT" "Checking optional MuDIS input path"
if [[ -f "${MUDIS_INPUT:-muonDis.root}" ]]; then
  echo "[OK] ${MUDIS_INPUT:-muonDis.root}"
else
  echo "[WARN] ${MUDIS_INPUT:-muonDis.root} not found"
fi

label "PREFLIGHT" "Checking Python interpreter"
if command -v python >/dev/null 2>&1; then
  python --version
else
  echo "[FAIL] python executable not found in PATH"
fi

label "PREFLIGHT" "Checking Python ROOT import"
python - <<'PY'
import importlib
try:
    importlib.import_module("ROOT")
    print("[OK] import ROOT")
except Exception as exc:
    print(f"[FAIL] import ROOT: {exc}")
PY

label "PREFLIGHT" "Checking run_simScript.py --help (read-only command)"
if [[ -n "${FAIRSHIP:-}" && -f "${FAIRSHIP}/macro/run_simScript.py" ]]; then
  python "${FAIRSHIP}/macro/run_simScript.py" --help || true
else
  echo "[SKIP] run_simScript.py --help (script not found)"
fi

label "PREFLIGHT" "Checking makeMuonDIS.py --help (read-only command)"
if [[ -n "${FAIRSHIP:-}" && -f "${FAIRSHIP}/muonDIS/makeMuonDIS.py" ]]; then
  python "${FAIRSHIP}/muonDIS/makeMuonDIS.py" --help || true
else
  echo "[SKIP] makeMuonDIS.py --help (script not found)"
fi
