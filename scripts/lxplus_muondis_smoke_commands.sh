#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/lxplus_muondis_smoke_commands.sh --print
  DY_VALUE=<float> bash scripts/lxplus_muondis_smoke_commands.sh --print
  DY_VALUE=<float> bash scripts/lxplus_muondis_smoke_commands.sh --execute
  USE_CODE_DEFAULT_DY=1 bash scripts/lxplus_muondis_smoke_commands.sh --print
  USE_CODE_DEFAULT_DY=1 bash scripts/lxplus_muondis_smoke_commands.sh --execute

Safety:
  - Default invocation prints this usage and exits.
  - No simulation command runs unless --execute is provided.
  - Choose exactly one DY mode:
      1) DY_VALUE=<float> (explicit value, printed as -Y "<value>")
      2) USE_CODE_DEFAULT_DY=1 (omit -Y, rely on code default 6.0 m)
  - If neither or both are set, MuonBack print/execute is refused.
  - DY is PROVISIONAL: parser/type/unit/default are code-backed; thesis-safe value/range is unresolved.
EOF
}

label() {
  printf '[%s] %s\n' "$1" "$2"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

MODE="$1"
if [[ "${MODE}" != "--print" && "${MODE}" != "--execute" ]]; then
  usage
  exit 1
fi

DY_VALUE_RAW="${DY_VALUE:-}"
USE_CODE_DEFAULT_DY_RAW="${USE_CODE_DEFAULT_DY:-}"

if [[ -n "${USE_CODE_DEFAULT_DY_RAW}" && "${USE_CODE_DEFAULT_DY_RAW}" != "1" ]]; then
  label "ERROR" "USE_CODE_DEFAULT_DY must be exactly 1 when set."
  exit 1
fi

if [[ -n "${DY_VALUE_RAW}" && "${USE_CODE_DEFAULT_DY_RAW}" == "1" ]]; then
  label "ERROR" "Set only one DY mode: DY_VALUE=<float> or USE_CODE_DEFAULT_DY=1."
  exit 1
fi

if [[ -z "${DY_VALUE_RAW}" && "${USE_CODE_DEFAULT_DY_RAW}" != "1" ]]; then
  label "UNRESOLVED" "Refusing MuonBack command: set DY_VALUE=<float> or USE_CODE_DEFAULT_DY=1."
  label "WARNING" "DY is PROVISIONAL: parser/type/unit/default are code-backed; thesis-safe value/range is unresolved."
  exit 1
fi

if [[ -n "${DY_VALUE_RAW}" ]]; then
  if ! [[ "${DY_VALUE_RAW}" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    label "ERROR" "DY_VALUE must be a float-compatible literal."
    exit 1
  fi
  MUONBACK_CMD='python "$FAIRSHIP/macro/run_simScript.py" --MuonBack -f "$SHIPSOFT/data/pythia8_Geant4_onlyMuons.root" -Y "'"${DY_VALUE_RAW}"'"'
  DY_MODE_NOTE='Using explicit DY_VALUE in meters via -Y "'"${DY_VALUE_RAW}"'".'
else
  MUONBACK_CMD='python "$FAIRSHIP/macro/run_simScript.py" --MuonBack -f "$SHIPSOFT/data/pythia8_Geant4_onlyMuons.root"'
  DY_MODE_NOTE='Using code default DY by omitting -Y (code default: 6.0 m).'
fi

MUDIS_CMD='python "$FAIRSHIP/macro/run_simScript.py" --MuDIS -f "${MUDIS_INPUT:-muonDis.root}" -i "${FIRST_EVENT:-0}" -n "${N_EVENTS:-1}"'
PROBE_CMD='python scripts/lxplus_muondis_oracle_probe.py --input "${PROBE_INPUT:-${MUDIS_INPUT:-muonDis.root}}" --output "${PROBE_OUTPUT:-oracle_probe.json}" --event-index "${EVENT_INDEX:-0}"'

label "WARNING" "DY is PROVISIONAL: parser/type/unit/default are code-backed; thesis-safe value/range is unresolved."
label "PROVISIONAL" "${DY_MODE_NOTE}"
label "CODE_ANCHORED" "${MUONBACK_CMD}"
label "PROVISIONAL" "${MUDIS_CMD}"
label "INSPECTION" "${PROBE_CMD}"
label "UNRESOLVED" "Canonical end-to-end MuonBack->MuDIS chain is NOT FOUND IN INDEX"

if [[ "${MODE}" == "--print" ]]; then
  exit 0
fi

label "CODE_ANCHORED" "Executing MuonBack candidate (explicit opt-in)"
eval "${MUONBACK_CMD}"

label "PROVISIONAL" "Executing MuDIS transport candidate (explicit opt-in)"
eval "${MUDIS_CMD}"

label "INSPECTION" "Executing oracle probe on existing ROOT output"
eval "${PROBE_CMD}"
