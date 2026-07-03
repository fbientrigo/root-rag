# FairShip Build Methods (pixi vs CVMFS + aliBuild)

**Audience:** anyone who needs a working **FairShip** installation to produce the
ROOT outputs that root-rag indexes and validates (Muon DIS workflow), or to point
`scripts/index_fairship.py` at a local FairShip checkout.

> **First, a clarification that avoids the most common confusion.**
> root-rag itself is a plain Python package — you install it with `pip install -e .`
> (see the top-level [README](../../../README.md)). It is **not** built with pixi,
> CVMFS, or aliBuild. Those tools build **FairShip**, which is a *separate* CERN
> project ([ShipSoft/FairShip](https://github.com/ShipSoft/FairShip)). You do not
> build root-rag on lxplus; you build/run FairShip there (or locally) and then use
> root-rag against its code and outputs.

FairShip offers **two** build paths upstream:

| | Method 1 — pixi | Method 2 — CVMFS + aliBuild |
|---|---|---|
| Status upstream | Recommended / modern | Maintenance mode / legacy |
| Where it runs | Any machine with pixi (local, laptop, CI) | lxplus or any host with CVMFS; or self-hosted with shipdist |
| What it produces | A reproducible FairShip environment from a lockfile | A compiled FairShip stack via aliBuild |
| Typical use | Day-to-day development, reproducible runs | Existing lxplus/CERN workflows, CI-validated releases |
| Env vars | `FAIRSHIP`, `GEOMPATH` auto-set to the clone | `FAIRSHIP`, `SHIPSOFT` set by the release/alienv |

---

## Method 1 — pixi (recommended)

[pixi](https://pixi.sh) resolves the full dependency stack from a lockfile, so the
build is reproducible across machines without CVMFS.

**Prerequisites**
- Install pixi: see <https://pixi.sh/latest/#installation>
- `git` (with `git-lfs` if you need the large data files)

**Build from source**
```bash
git clone https://github.com/ShipSoft/FairShip.git
cd FairShip
pixi run build
```

> **Heads-up:** the first `pixi run build` solves and downloads a large dependency
> environment (multiple GB) and compiles FairShip, so it can take a long time — it is
> not stalled. Subsequent runs reuse the cached environment.

**Verify the build succeeded** (should print the simulation CLI help, not an import
error):
```bash
pixi run python macro/run_simScript.py --help
```

**Run inside the environment** — either invoke per-command or open a shell:
```bash
# one-off command
pixi run python macro/run_simScript.py --tag my-simulation

# or an interactive shell with the environment activated
pixi shell
python macro/run_simScript.py --tag my-simulation
```

**Using pre-built packages** (skip compiling from source):
```bash
git clone https://github.com/ShipSoft/FairShip.git
cd FairShip
pixi add fairship
pixi run python macro/run_simScript.py --tag my-simulation
```

**Important:** the pixi project root must be the FairShip clone itself — it carries
the required `geometry/` and `files/` directories, and pixi sets `FAIRSHIP` and
`GEOMPATH` to `PIXI_PROJECT_ROOT` automatically.

> Commands above mirror the current upstream FairShip README. If a step changes,
> defer to <https://github.com/ShipSoft/FairShip> as the source of truth.

---

## Method 2 — CVMFS + aliBuild (legacy / lxplus)

This is the path most existing SHiP/lxplus workflows use. FairShip is built with
aliBuild against a CVMFS-provided software release.

**Environment variables used below**
- `$SHIP_RELEASE` — the CVMFS software release you build against. Pick a valid value
  from the [`ShipSoft/cvmfs_release`](https://github.com/ShipSoft/cvmfs_release) repo;
  root-rag records validated releases in
  [`configs/support_matrix.yaml`](../../../configs/support_matrix.yaml)
  (e.g. `26.02` / `26.03`, `slc9_x86-64`).
- `$SHIPDIST` — the path to the FairShip build recipes; it is **exported for you when
  you `source setUp.sh`** on CVMFS. (Self-hosted builds set it to a local `shipdist`
  clone — see below.)

**On lxplus (or any host with CVMFS):**
```bash
# 1. Clone (git-lfs pulls the large data files)
git lfs install
git clone https://github.com/ShipSoft/FairShip.git

# 2. Verify CVMFS is mounted
ls /cvmfs/ship.cern.ch

# 3. Source a release (note the capital U in setUp.sh — this is the canonical name).
#    This also exports $SHIPDIST used in the next step.
source /cvmfs/ship.cern.ch/$SHIP_RELEASE/setUp.sh

# 4. Build FairShip with aliBuild
aliBuild build FairShip --always-prefer-system --config-dir $SHIPDIST --defaults release

# 5. Load the environment
alienv enter FairShip/latest
# non-interactive equivalent:
eval $(alienv load FairShip/latest --no-refresh)
```

> **Heads-up:** the aliBuild step compiles a multi-GB software stack and the first
> build can take a long time (tens of minutes or more). It is not stalled.

**Verify the build succeeded** (inside the loaded environment):
```bash
python "$FAIRSHIP/macro/run_simScript.py" --help
```
This is the same probe that [`scripts/lxplus_muondis_preflight.sh`](../../../scripts/lxplus_muondis_preflight.sh)
runs — if it prints the CLI help rather than an import/symbol error, the environment
is loaded correctly.

**Self-hosted (no CVMFS):**
```bash
# install aliBuild via pipx or pip, then provide the recipes via shipdist
git clone https://github.com/ShipSoft/shipdist.git
aliBuild build FairShip --config-dir ./shipdist --defaults release
alienv enter FairShip/latest
```

Legacy branches use `--defaults fairship-2018` instead of `--defaults release`.

> Validated CVMFS releases and OS targets for this repo's anchor commit live in
> [`configs/support_matrix.yaml`](../../../configs/support_matrix.yaml).

---

## Which method should I use?

- **You want reproducibility / are off-CERN / on a laptop or CI** → **pixi**.
- **You are on lxplus or already have a CVMFS-based SHiP workflow** → **CVMFS + aliBuild**.
- Both produce the same FairShip; choose by environment, not by capability.

### Troubleshooting "I can't build FairShip on lxplus"

- You do **not** build *root-rag* on lxplus — only FairShip. root-rag is
  `pip install -e .` on any machine.
- If `aliBuild` fails, confirm CVMFS is mounted (`ls /cvmfs/ship.cern.ch`) and that
  you sourced a valid `$SHIP_RELEASE` setup before building.
- `TClass`/`undefined symbol` errors at runtime usually mean a mismatch between the
  loaded release and the local checkout — re-`alienv load` a matching release.
- If aliBuild itself is the blocker, the **pixi** path sidesteps CVMFS/aliBuild
  entirely and is the recommended fallback.

---

## How each method connects to root-rag

Once FairShip is available by either method:

1. **Index the FairShip source** (point at your local clone):
   ```bash
   python scripts/index_fairship.py --fairship-path /path/to/FairShip
   ```
2. **Run the Muon DIS simulation** to produce ROOT outputs (inside the FairShip
   environment), e.g. `run_simScript.py --MuDIS` / `--MuonBack`.
3. **Validate on lxplus** with the preflight + oracle-probe helpers:
   - [`scripts/lxplus_muondis_preflight.sh`](../../../scripts/lxplus_muondis_preflight.sh)
     (checks `FAIRSHIP`, `SHIPSOFT`, `run_simScript.py`, `makeMuonDIS.py`)
   - [`scripts/lxplus_muondis_oracle_probe.py`](../../../scripts/lxplus_muondis_oracle_probe.py)
     (inspects the produced ROOT file)

## See also

- [LXPLUS preflight checklist](LXPLUS_preflight_checklist.md)
- [How to validate MuonDIS on LXPLUS](How_to_validate_MuonDIS_on_LXPLUS.md)
- [FairShip Muon DIS workflow](../../llm_wiki/fairship_muondis_workflow.md)
- Upstream source of truth: <https://github.com/ShipSoft/FairShip>
