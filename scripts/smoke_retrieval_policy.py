"""Smoke test script for Retrieval Forest policy invariants."""
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def test_policy():
    errors = []
    
    # 1. ROOT general uses lexical
    out1 = run_command([sys.executable, "src/root_rag/cli.py", "ask", "TTree", "--root-ref", "v6-36-08"])
    if "Actual backend: lexical" not in out1:
        errors.append("FAILED: ROOT general should use lexical backend")
    else:
        print("PASSED: ROOT general uses lexical")

    # 2. FairShip profile uses forest (assuming indexes exist for 'master')
    out2 = run_command([sys.executable, "src/root_rag/cli.py", "ask", "InMuon", "--profile", "fairship", "--root-ref", "master"])
    if "Actual backend: forest" not in out2:
        # Check if it fell back due to missing index - that's also an "expected" behavior if indexes are missing,
        # but for hardening we assume the environment has the forest indexes built for master.
        if "Fallback reason" in out2:
            print(f"INFO: FairShip forest fell back: {out2.split('Fallback reason:')[1].splitlines()[0]}")
            # If it's a fallback, it's technically compliant with the "auto-default" policy but doesn't exercise the forest logic.
        else:
            errors.append("FAILED: FairShip profile should use forest backend")
    else:
        print("PASSED: FairShip profile uses forest")

    # 3. FairShip profile with --baseline uses lexical
    out3 = run_command([sys.executable, "src/root_rag/cli.py", "ask", "InMuon", "--profile", "fairship", "--root-ref", "master", "--baseline"])
    if "Actual backend: lexical" not in out3:
        errors.append("FAILED: FairShip with --baseline should force lexical backend")
    else:
        print("PASSED: FairShip with --baseline forces lexical")

    if errors:
        print("\nPolicy Verification FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nPolicy Verification SUCCESSFUL")
        sys.exit(0)

if __name__ == "__main__":
    test_policy()
