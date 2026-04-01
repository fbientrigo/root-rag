# ROOT-RAG How-To Guide
**Complete guide for using ROOT-RAG effectively**

---

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Queries](#advanced-queries)
4. [Building Indices](#building-indices)
5. [Understanding Results](#understanding-results)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
```bash
# Required
Python 3.10+
Git 2.30+

# Optional (for development)
pytest 9.0+
```

### Quick Install
```bash
# Clone repository
git clone https://github.com/fbientrigo/root-rag
cd root-rag

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
root-rag --help
```

### First-Time Setup
```bash
# Test with existing indices (instant)
root-rag ask "TTree Fill"

# If no indices exist, build Tier 1 (one-time, ~45 seconds)
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/tier1_corpus_root_636.yaml \
  --output-dir data/indexes_tier1
```

---

## Basic Usage

### Query Patterns

#### 1. Find API Definitions
```bash
# Class methods
root-rag ask "TTree Fill method"
root-rag ask "TGeoManager MakeBox"
root-rag ask "TVector3 SetXYZ"

# Function signatures
root-rag grep "virtual Int_t Fill"
root-rag grep "TGeoVolume* MakeBox"

# Constants and enums
root-rag ask "kRed kBlue kGreen"
root-rag grep "enum EColor"
```

#### 2. Find Usage Examples
```bash
# FairShip patterns
root-rag ask "DetectorHit ProcessHits FairShip"
root-rag ask "TGeoManager AddNode FairShip geometry"
root-rag ask "TTree Branch AddBranch FairShip"

# Cross-reference ROOT + FairShip
root-rag ask "TVector3 momentum FairShip"
root-rag ask "TLorentzVector energy mass"
```

#### 3. Explore SOFIE
```bash
# Find operators
root-rag grep "ROperator_Conv"
root-rag grep "ROperator_Relu"

# Understand workflow
root-rag ask "RModel Generate ONNX"
root-rag ask "SOFIE operator interface"
```

---

## Advanced Queries

### Using Flags

#### Limit Results
```bash
# Top 5 results
root-rag ask "TTree Fill" --top-k 5

# Top 20 results
root-rag grep "TGeoManager" --top-k 20
```

#### Search Specific Index
```bash
# Search only ROOT Tier 1
root-rag grep "TTree::Fill" --index tier1

# Search only FairShip
root-rag grep "DetectorHit" --index fairship

# Search only SOFIE
root-rag grep "ROperator" --index sofie
```

#### JSON Output
```bash
# Machine-readable output
root-rag ask "TTree Fill" --json

# Save to file
root-rag ask "TTree Fill" --json > results.json

# Pipe to jq
root-rag ask "TTree Fill" --json | jq '.evidence[0]'
```

### Query Optimization

#### ✅ Good Queries (Keyword-Based)
```bash
root-rag ask "TTree Fill Branch"          # ✓ Clear keywords
root-rag ask "TGeoManager MakeBox AddNode" # ✓ API methods
root-rag ask "TVector3 momentum physics"   # ✓ Domain terms
```

#### ❌ Bad Queries (Natural Language)
```bash
root-rag ask "How do I fill a TTree?"          # ✗ Too conversational
root-rag ask "What's the best way to use...?"  # ✗ Subjective
root-rag ask "Can you explain TGeoManager?"    # ✗ Too broad
```

**Why?** ROOT-RAG uses lexical BM25 search (like grep, not ChatGPT). Keywords work better than questions.

---

## Building Indices

### Index ROOT Corpus

#### Tier 1 (Recommended)
```bash
# 35 most-used ROOT classes
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/tier1_corpus_root_636.yaml \
  --output-dir data/indexes_tier1

# Result: 1,106 chunks from 53 files (~45 seconds)
```

#### SOFIE Operators
```bash
# 39 ML inference operators
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/sofie_corpus_root_636.yaml \
  --output-dir data/indexes_sofie

# Result: 140 chunks from 40 files (~10 seconds)
```

#### Custom Corpus
```bash
# Create corpus config (YAML)
cat > configs/my_corpus.yaml << 'EOF'
root:
  version: "6.36.08"
  tag: "v6-36-08"
  repository: "root-project/root"

corpus:
  tier: "custom"
  classes:
    - name: "MyClass"
      headers:
        - "path/to/MyClass.h"
      sources:
        - "path/to/MyClass.cxx"
EOF

# Index it
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/my_corpus.yaml \
  --output-dir data/indexes_custom
```

### Index FairShip Codebase

```bash
# Requires local FairShip clone
git clone https://github.com/ShipSoft/FairShip ../FairShip

# Run indexing script
python scripts/index_fairship.py --fairship-path ../FairShip

# Result: 386 chunks from 163 files (~30 seconds)
```

### List Available Indices
```bash
root-rag versions

# Output:
# Available Indices:
# 
# ROOT Tier 1 (v6-36-08):
#   - Index ID: v6-36-08__9005eb7d69f1__20260331T181700091086+0000Z
#   - Chunks: 1,106
#   - Files: 53
#   - Created: 2026-03-31
# 
# FairShip (master):
#   - Index ID: fairship__master__98de16a5b264__20260331T185059271533+0000Z
#   - Chunks: 386
#   - Files: 163
#   - Created: 2026-03-31
# 
# SOFIE (v6-36-08):
#   - Index ID: v6-36-08__9005eb7d69f1__20260401T155801865132+0000Z
#   - Chunks: 140
#   - Files: 40
#   - Created: 2026-04-01
```

---

## Understanding Results

### Result Format

```bash
$ root-rag ask "TTree Fill"
```

**Output:**
```
Evidence (ROOT v6-36-08, commit 9005eb7d69f1):

[1] tree/tree/inc/TTree.h:234-289 (score: 15.3)
    virtual Int_t Fill();
    // Fill all branches. Returns number of bytes written to file.
    
[2] tree/tree/src/TTree.cxx:567-612 (score: 12.1)
    Int_t TTree::Fill() {
        // Implementation details...
    }

Source Attribution:
  - ROOT Tier 1 (v6-36-08): 2 results
```

### Result Components

1. **Version Tag**: `v6-36-08, commit 9005eb7d69f1`
   - Ensures reproducibility
   - No version mixing

2. **File Path**: `tree/tree/inc/TTree.h`
   - Relative to ROOT repository root
   - POSIX-style paths (forward slashes)

3. **Line Range**: `:234-289`
   - 1-indexed, inclusive
   - Can navigate to exact location

4. **Score**: `(score: 15.3)`
   - BM25 relevance score
   - Higher = more relevant
   - Scores can be negative (BM25 algorithm)

5. **Content Snippet**
   - Exact code from file (no modifications)
   - Preserves formatting, comments

6. **Source Attribution**
   - Which index provided the result
   - How many results from each source

---

## Tips & Best Practices

### Query Strategy

1. **Start Broad, Then Narrow**
   ```bash
   # Broad
   root-rag ask "TTree"
   
   # Narrower
   root-rag ask "TTree Fill"
   
   # Specific
   root-rag ask "TTree Fill Branch AddBranch"
   ```

2. **Use Class::Method Format**
   ```bash
   # ✓ Good
   root-rag ask "TTree::Fill"
   root-rag ask "TGeoManager::MakeBox"
   
   # ✗ Less effective
   root-rag ask "Fill method in TTree"
   ```

3. **Combine ROOT + FairShip**
   ```bash
   # Find API + usage together
   root-rag ask "TGeoManager MakeBox FairShip"
   root-rag ask "TVector3 momentum FairShip physics"
   ```

### When to Use `ask` vs `grep`

**Use `ask`:**
- Natural queries: "TTree Fill method"
- Cross-index search (ROOT + FairShip)
- Want ranked results with scores
- Need JSON output

**Use `grep`:**
- Exact string match: "virtual Int_t Fill"
- Fast keyword search
- Know exactly what you're looking for
- Simple text pattern matching

### Performance Tips

1. **Cache-Friendly Queries**
   ```bash
   # FTS5 caches recent queries
   # Repeat queries are instant
   root-rag ask "TTree Fill"  # First time: 100ms
   root-rag ask "TTree Fill"  # Second time: <10ms
   ```

2. **Limit Results for Speed**
   ```bash
   # Top 5 is faster than top 50
   root-rag ask "TTree" --top-k 5
   ```

3. **Use Specific Index**
   ```bash
   # Faster than searching all indices
   root-rag grep "TTree::Fill" --index tier1
   ```

---

## Troubleshooting

### Issue: "Index not found"

**Problem:**
```bash
$ root-rag ask "TTree Fill"
Error: Index not found for ROOT version v6-36-08
```

**Solution:**
```bash
# Build the index first
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/tier1_corpus_root_636.yaml \
  --output-dir data/indexes_tier1

# Then query
root-rag ask "TTree Fill"
```

---

### Issue: "fts5: syntax error near '/'"

**Problem:**
```bash
$ root-rag ask "How do I use TTree::Fill?"
Error: fts5: syntax error near '/'
```

**Solution:**
FTS5 has special characters (`/`, `?`, `:`, etc.). Use keywords instead:
```bash
# ✗ Bad
root-rag ask "How do I use TTree::Fill?"

# ✓ Good
root-rag ask "TTree Fill method"
```

---

### Issue: "No results found"

**Problem:**
```bash
$ root-rag ask "MyCustomClass"
No evidence found.
```

**Possible Causes:**

1. **Class not in corpus**
   ```bash
   # Check what's indexed
   root-rag versions
   
   # Add to custom corpus if needed
   # (see "Building Indices" → "Custom Corpus")
   ```

2. **Typo in query**
   ```bash
   # ✗ Wrong
   root-rag ask "TTree::Fil"
   
   # ✓ Correct
   root-rag ask "TTree::Fill"
   ```

3. **Need broader keywords**
   ```bash
   # ✗ Too specific
   root-rag ask "TTree Fill with weighted entries and buffer management"
   
   # ✓ Better
   root-rag ask "TTree Fill"
   ```

---

### Issue: "Results not relevant"

**Problem:**
Results don't match what you're looking for.

**Solution:**

1. **Refine keywords**
   ```bash
   # Add more context
   root-rag ask "TTree Fill Branch AddBranch"
   ```

2. **Use cross-index search**
   ```bash
   # Search ROOT + FairShip together
   root-rag ask "TGeoManager MakeBox FairShip"
   ```

3. **Check multiple queries**
   ```bash
   # Try variations
   root-rag ask "TTree Fill"
   root-rag ask "TTree Branch"
   root-rag ask "TTree Write"
   ```

---

### Issue: "Slow indexing"

**Problem:**
Indexing takes too long.

**Expected Times:**
- Tier 1 (53 files): ~45 seconds
- FairShip (163 files): ~30 seconds
- SOFIE (40 files): ~10 seconds

**If slower:**

1. **Check disk I/O**
   ```bash
   # Indexing is I/O-bound
   # SSD vs HDD makes a difference
   ```

2. **Reduce corpus size**
   ```bash
   # Start with fewer classes
   # Build incrementally
   ```

3. **Check ROOT clone size**
   ```bash
   # First-time indexing clones ROOT (~1 GB)
   # Subsequent builds use cache
   du -sh data/raw/corpora/
   ```

---

## Advanced Topics

### Corpus Design

**Corpus YAML Structure:**
```yaml
root:
  version: "6.36.08"
  tag: "v6-36-08"
  repository: "root-project/root"

corpus:
  tier: "my_tier"
  rationale: "Why these classes?"
  
  classes:
    - name: "ClassName"
      rationale: "Why include this?"
      headers:
        - "path/to/header.h"
      sources:
        - "path/to/source.cxx"
```

**Best Practices:**
- Keep corpus focused (10-50 classes)
- Document rationale for each class
- Test with golden queries
- Start small, expand iteratively

### Golden Queries

Golden queries validate retrieval quality:

```json
{
  "query_id": "gq_ttree_fill",
  "query": "TTree Fill",
  "natural_language": "Where is TTree::Fill defined?",
  "category": "root_io",
  "expected_evidence": {
    "min_results": 2,
    "must_contain_files": [
      "tree/tree/inc/TTree.h",
      "tree/tree/src/TTree.cxx"
    ]
  }
}
```

**Add golden queries in:**
- `configs/tier1_golden_queries.json` (ROOT Tier 1)
- `configs/fairship_golden_queries.json` (FairShip)
- `configs/sofie_golden_queries.json` (SOFIE - future)

---

## Summary

**Quick Reference:**

| Task | Command |
|------|---------|
| Query existing indices | `root-rag ask "query"` |
| Fast keyword search | `root-rag grep "keyword"` |
| List available indices | `root-rag versions` |
| Build ROOT index | `root-rag index --root-ref v6-36-08 --seed-corpus configs/tier1_corpus_root_636.yaml` |
| Build FairShip index | `python scripts/index_fairship.py --fairship-path ../FairShip` |
| Get help | `root-rag --help` |

**Remember:**
- ✅ Use keywords, not questions
- ✅ Start broad, then narrow
- ✅ Combine ROOT + FairShip for context
- ✅ Check `root-rag versions` to see what's available
- ✅ Build indices before querying (one-time setup)

**Need help?** Check [`docs/`](../docs/) or open a [GitHub issue](https://github.com/fbientrigo/root-rag/issues).
