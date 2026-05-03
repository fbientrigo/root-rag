# 00 Environment

Use single-line PowerShell commands only. Start from repository root in conda env `rootrag`.

## Primary Path

```powershell
python scripts/emv_preflight.py
```

```powershell
python scripts/emv_status.py
```

## Manual Fallback Bundle

```powershell
conda activate rootrag
```

```powershell
Get-Location
```

```powershell
python -c "import sys; print(sys.executable)"
```

```powershell
python -c "import yaml; print(yaml.__version__)"
```

```powershell
root-rag ask --help
```

```powershell
root-rag search --help
```

```powershell
Get-ChildItem data/indexes_fairship -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,FullName
```

## Stop Conditions

- If PyYAML import fails, stop before running query packs.
- If `root-rag ask --help` fails, stop before claiming CLI execution is healthy.
- If `root-rag search --help` fails, stop before claiming retrieval CLI coverage is healthy.
- If `data/indexes_fairship` is missing or has no valid child index with `fts.sqlite` or `index_manifest.json`, stop before running Muon DIS smoke commands.
