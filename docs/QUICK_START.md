# Quick Reference: FairShip ROOT Usage Extraction

## One-Line Command

```bash
python scripts/extract_fairship_root_usage.py --fairship-path <path-to-fairship>
```

## Example with Real FairShip

```bash
# If FairShip is in a sibling directory
python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip

# If FairShip is elsewhere
python scripts/extract_fairship_root_usage.py --fairship-path /home/user/projects/FairShip
```

## Example with Mock Data (Testing)

```bash
# Step 1: Create mock FairShip
python scripts/create_mock_fairship.py

# Step 2: Use the path it prints
python scripts/extract_fairship_root_usage.py --fairship-path <printed-path>
```

## Custom Output Paths

```bash
python scripts/extract_fairship_root_usage.py \
  --fairship-path ../FairShip \
  --json-output my_results/inventory.json \
  --markdown-output my_results/report.md
```

## Default Output Locations

- **JSON**: `artifacts/fairship_root_usage_inventory.json`
- **Markdown**: `reports/fairship_root_usage_inventory.md`

## Run Tests

```bash
python tests/test_extract_fairship.py
```

## View Help

```bash
python scripts/extract_fairship_root_usage.py --help
```

## What You Get

### JSON Artifact
- Scanned/matched file counts
- Complete list of ROOT headers with usage counts
- Complete list of ROOT symbols with usage counts
- Per-module summary statistics

### Markdown Report
- Executive summary
- Top 20 ROOT headers (ranked)
- Top 20 ROOT symbols (ranked)
- Usage by FairShip module
- Methodology notes
- Suggested next steps

## Expected Runtime

- Small projects (~100 files): < 5 seconds
- Medium projects (~1000 files): < 30 seconds
- Large projects (FairShip scale): 1-3 minutes

## Requirements

- Python 3.9+
- No external dependencies
- Read access to FairShip directory

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "FairShip path does not exist" | Check the path is correct |
| "No files scanned" | Ensure FairShip has .cxx/.h files |
| "Permission denied" | Check read permissions |
| Python not found | Use `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Unix) |

## Key Features

✅ Deterministic (same input → same output)
✅ Conservative filtering (low false positive rate)
✅ Fast (regex-based, no AST parsing)
✅ Comprehensive (headers + symbols + modules)
✅ Evidence-based (shows which files use what)

## Next Steps After Extraction

1. Review `reports/fairship_root_usage_inventory.md`
2. Identify top 20-30 most-used features (Tier 1)
3. Identify heavily-used subsystems (Tier 2)
4. Use findings to prioritize ROOT documentation corpus
5. Update RAG retrieval ranking based on evidence

## Documentation

- Full guide: `docs/fairship_extraction.md`
- Implementation summary: `reports/T1_implementation_summary.md`
- Test suite: `tests/test_extract_fairship.py`
