# FairShip ROOT Usage Extraction

## Overview

This tool automatically extracts evidence of ROOT usage from a local FairShip clone. It performs regex-based scanning to identify ROOT headers and symbols, generating both machine-readable (JSON) and human-readable (Markdown) reports.

## Purpose

The extraction serves as evidence-based input for:
1. **Corpus prioritization**: Identifying which ROOT documentation is most relevant
2. **Tier 1/Tier 2 derivation**: Determining critical vs. supporting ROOT components
3. **RAG optimization**: Focusing retrieval on actually-used ROOT features

## Usage

### Basic Command

```bash
python scripts/extract_fairship_root_usage.py --fairship-path <path-to-fairship>
```

### Example

```bash
python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip
```

### Full Options

```bash
python scripts/extract_fairship_root_usage.py \
  --fairship-path ../FairShip \
  --json-output artifacts/fairship_root_usage_inventory.json \
  --markdown-output reports/fairship_root_usage_inventory.md
```

## Outputs

### 1. JSON Artifact
**Location**: `artifacts/fairship_root_usage_inventory.json`

Machine-readable structured data including:
- Total counts (scanned files, matched files, headers, symbols)
- Complete list of ROOT headers with usage counts and file locations
- Complete list of ROOT symbols with usage counts and file locations
- Per-module summary with top headers/symbols

**Use case**: Programmatic analysis, further processing, visualization

### 2. Markdown Report
**Location**: `reports/fairship_root_usage_inventory.md`

Human-readable summary including:
- Executive summary with key statistics
- Top 20 ROOT headers (ranked by usage)
- Top 20 ROOT symbols (ranked by usage)
- Usage breakdown by FairShip module
- Methodology notes and limitations
- Suggested next steps

**Use case**: Review, validation, decision-making

## What is Extracted

### ROOT Headers
Patterns matched:
- `#include "T*.h"` (e.g., TFile.h, TTree.h)
- `#include <ROOT/*.h>` (e.g., ROOT/RDataFrame.hxx)
- `#include "TMVA/*.h"` (e.g., TMVA/Reader.h)
- `#include "Math/*.h"` (e.g., Math/Vector3D.h)
- Common ROOT headers (Rtypes.h, RVersion.h, etc.)

### ROOT Symbols
Patterns matched:
- T-prefixed classes (TFile, TTree, TH1F, TGeoManager, etc.)
- ROOT namespace references (ROOT::Math::*, ROOT::RDataFrame, etc.)
- TMVA namespace references (TMVA::Reader, etc.)
- Math namespace references

**Filtering**: The tool applies conservative filtering to reduce false positives while maintaining high precision for common ROOT symbols.

## Methodology

### Scanning Strategy
1. Recursively scan all source files in FairShip
2. File extensions: `.cxx`, `.cpp`, `.cc`, `.h`, `.hpp`, `.hh`, `.C`, `.py`
3. Ignored directories: `.git`, `build`, `cmake-build*`, `dist`, `__pycache__`, `external`

### Pattern Matching
- **Approach**: Regex-based text analysis (no AST parsing)
- **Rationale**: Balance between speed, simplicity, and accuracy
- **Trade-offs**: Accepts some false positives for comprehensive coverage

### Module Grouping
Files are grouped by their top-level directory in FairShip, enabling analysis of which components use which ROOT features most heavily.

## Limitations

1. **No semantic analysis**: Text-based matching, not semantic parsing
2. **False positives**: Some T-prefixed symbols may not be ROOT classes
3. **False negatives**: May miss usage in macros, generated code, or dynamic features
4. **Count methodology**: Counts file occurrences, not in-file usage frequency

## Confidence Levels

- **ROOT headers**: **HIGH** (direct #include detection)
- **Common symbols** (TFile, TTree, TH1, TGeo*, etc.): **HIGH**
- **Generic T-prefixed symbols**: **MEDIUM** (likely ROOT but not guaranteed)
- **Namespace qualifiers**: **HIGH** (ROOT::, TMVA::, Math::)

## Next Steps

After running this extraction:

1. **Review the reports** in `reports/fairship_root_usage_inventory.md`
2. **Validate top findings** (spot-check high-frequency headers/symbols)
3. **Derive Tier 1/Tier 2**:
   - Tier 1: Top 20-30 most-used features (critical documentation)
   - Tier 2: Heavily-used subsystems (e.g., TGeo*, TMVA, Math)
4. **Update RAG ranking** based on evidence

## Testing

Run the validation tests:

```bash
python tests/test_extract_fairship.py
```

Tests cover:
- Header extraction accuracy
- Symbol filtering effectiveness
- Module grouping correctness
- JSON generation validity

## Requirements

- Python 3.9+
- Standard library only (no external dependencies)
- Local FairShip clone

## Design Principles

1. **Deterministic**: Same input always produces same output
2. **Modular**: Clear separation of scanning, extraction, and reporting
3. **Typed**: Uses dataclasses and type hints for clarity
4. **Lightweight**: No heavy dependencies
5. **Reviewable**: Clear code structure for validation

## Architecture

```
FairShipROOTExtractor
├── scan_directory()      # Recursively scan FairShip files
├── scan_file()           # Process individual file
├── _extract_includes()   # Find ROOT headers
├── _extract_symbols()    # Find ROOT symbols
├── _is_likely_root_symbol()  # Filter false positives
├── generate_json_report()    # Create JSON artifact
└── generate_markdown_report() # Create Markdown report
```

## Example Output

### Console
```
Scanning FairShip at: /path/to/FairShip
Scanned 2,341 files, found ROOT usage in 1,847 files
JSON report written to: artifacts/fairship_root_usage_inventory.json
Markdown report written to: reports/fairship_root_usage_inventory.md

Extraction complete!
```

### Top Headers (sample)
1. TObject.h - 523 files
2. TFile.h - 412 files
3. TTree.h - 387 files
4. TGeoManager.h - 298 files
5. TH1.h - 245 files

### Top Symbols (sample)
1. TObject - 521 files
2. TFile - 409 files
3. TTree - 385 files
4. TGeoManager - 295 files
5. TVector3 - 234 files

## Troubleshooting

### "FairShip path does not exist"
Ensure you're providing the correct path to your local FairShip clone.

### "No files scanned"
Check that the FairShip directory contains source files with recognized extensions.

### "Permission denied"
Ensure read permissions on the FairShip directory.

## Contributing

To extend the extraction:
1. Add new regex patterns to `_compile_patterns()`
2. Update filtering logic in `_is_likely_root_symbol()`
3. Add tests to `tests/test_extract_fairship.py`
4. Run tests to validate changes

## License

MIT (same as root-rag project)
