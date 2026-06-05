"""Render a compact LXPLUS MuonDIS run checklist."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    category: str
    command: str
    note: str


STEPS = [
    Step("PREFLIGHT", 'echo "FAIRSHIP=${FAIRSHIP:-UNSET}"', "Verify FAIRSHIP is set."),
    Step("PREFLIGHT", 'echo "SHIPSOFT=${SHIPSOFT:-UNSET}"', "Verify SHIPSOFT is set."),
    Step("PREFLIGHT", 'test -f "$FAIRSHIP/macro/run_simScript.py"', "Confirm run_simScript.py exists."),
    Step("PREFLIGHT", 'test -f "$FAIRSHIP/muonDIS/makeMuonDIS.py"', "Confirm makeMuonDIS.py exists."),
    Step(
        "CODE_ANCHORED",
        'python "$FAIRSHIP/macro/run_simScript.py" --MuonBack -f "$SHIPSOFT/data/pythia8_Geant4_onlyMuons.root" -Y <DY>',
        "Code-anchored fragment; not yet validated.",
    ),
    Step("PROVISIONAL", 'python "$FAIRSHIP/macro/run_simScript.py" --MuDIS -f "<muonDis.root>" -i <firstEvent> -n <nEvents>', "Candidate MuDIS transport pattern."),
    Step("INSPECTION", "python <inspect_dis_branches.py>", "Check DIS tree and required branches."),
    Step("INSPECTION", "python <inspect_cbmsim.py> <transport_output.root>", "Check cbmsim and CrossSection branch."),
    Step("CLEANUP", "rm -f lxplus_muondis_preflight.log lxplus_muondis_inspection.log", "Optional local cleanup."),
    Step("UNRESOLVED", "NOT FOUND IN INDEX", "Canonical one-command chain remains unresolved."),
]


def main() -> int:
    print("# LXPLUS MuonDIS Checklist")
    print()
    for idx, step in enumerate(STEPS, start=1):
        print(f"{idx}. [{step.category}] `{step.command}`")
        print(f"   - {step.note}")
    print()
    print("## Log Template")
    print("- Command class:")
    print("- Command executed:")
    print("- Input files:")
    print("- Output file:")
    print("- Trees found:")
    print("- Branch counts:")
    print("- Errors:")
    print("- Interpretation:")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
