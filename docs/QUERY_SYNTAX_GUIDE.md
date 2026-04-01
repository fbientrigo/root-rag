# Query Syntax Guide for root-rag

This guide explains how queries work in `root-rag ask` and `root-rag grep` commands, including which words are filtered and how to write effective queries.

## Overview

ROOT-RAG uses **lexical BM25 search** (SQLite FTS5) under the hood. This means:
- Queries are **keyword-based**, not natural language
- **Stop words** (low-signal terms) are automatically filtered out
- **Exact matches** and **phrase searches** work best
- Query order **does not matter** (search is symmetric)

## Stop Words (Filtered Terms)

The following common words are automatically removed from your query:

```
and, in, through, with, files, file, fairship, root, usage, pattern, 
implementation, implementations, overrides, override, modules, module, 
detectors, detector, definition, test, validation, global, local, object
```

### Why These Words Are Filtered

- **Generic terms**: "and", "in", "with" → noise in code search
- **Meta-discussion**: "pattern", "implementation", "usage" → too broad
- **Project names**: "root", "fairship" → implicit in search context
- **Test/debug**: "test", "validation" → not code content

### Impact on Your Queries

| Query | After Filtering | Result |
|-------|-----------------|--------|
| `"Where is TGeoManager declared?"` | `"tgeomanager"` | ✅ 1 result (sparse) |
| `"Where is TGeoManager?"` | `"tgeomanager"` | ✅ 5 results (better) |
| `"Where TGeoManager?"` | `"tgeomanager"` | ✅ 5 results (same) |
| `"I'm searching for TGeoManager"` | `"searching" "tgeomanager"` | ❌ 0 results (no match) |

**Key Insight**: "Im searchin for TGeoManager?" becomes `["searching", "tgeomanager"]` after tokenization. Since "searching" has a typo and isn't in the index, it returns 0 results. The word "for" is filtered as a stop word.

## Query Construction Best Practices

### ✅ Do: Use Keywords

Keep queries simple and focused on **symbols, class names, and method names**:

```bash
root-rag ask "TTree::Fill"
root-rag ask "TGeoManager"
root-rag ask "TVector3 magnitude"
```

**Why**: BM25 scoring matches exact terms in indexed code chunks.

### ✅ Do: Use Multiple Keywords

Combine terms to narrow results:

```bash
root-rag ask "TTree Fill Branch"      # Find TTree's Fill and Branch methods
root-rag ask "TGeoManager MakeBox"    # Find geometry creation
root-rag ask "TVector3 momentum"      # Find physics calculations
```

**Why**: More specific queries return more relevant chunks.

### ✅ Do: Be Precise

Use exact class/method names when possible:

```bash
root-rag ask "ProcessHits DetectorHit"  # FairShip detector pattern
root-rag ask "RDataFrame Filter"        # Data analysis API
```

### ✅ Do: Mix Across Codebases

ROOT-RAG searches ROOT + FairShip + SOFIE simultaneously:

```bash
root-rag ask "TTree Fill FairShip"    # ROOT API + FairShip usage
root-rag ask "TGeoManager detector"   # ROOT geometry + FairShip detectors
```

### ❌ Don't: Use Natural Language

Avoid full sentences or conversational queries:

```bash
# ❌ Bad: "How do I create a histogram in ROOT?"
# ✅ Good: "TH1F histogram creation"

# ❌ Bad: "I need to understand TTree branches"
# ✅ Good: "TTree Branch"

# ❌ Bad: "What is the definition of Fill?"
# ✅ Good: "TTree::Fill"
```

**Why**: Sentences contain stop words that get filtered, and FTS5 doesn't understand natural language semantics.

### ❌ Don't: Include Stop Words in Key Positions

Avoid placing stop words where they block important terms:

```bash
# ❌ Weak: "What is the definition of TTree?" → becomes "ttree"
# ✅ Better: "TTree Fill"

# ❌ Weak: "Usage patterns with TGeoManager" → becomes "tgeomanager"
# ✅ Better: "TGeoManager AddNode"
```

### ❌ Don't: Use Typos

Misspelled keywords return 0 results:

```bash
# ❌ "Im searchin for TGeoManager" → ["searching", "tgeomanager"] → no match
# ✅ "Searching for TGeoManager" → ["searching", "tgeomanager"] → possible match
# ✅ "TGeoManager" → ["tgeomanager"] → definite match
```

**Tip**: For typo-tolerant search, just use the correct spelling of the class/method name.

## Query Alias Expansions (Advanced)

ROOT-RAG includes **intelligent alias expansion** for common ROOT patterns. When you search for certain terms, the system automatically expands them to related keywords:

### Supported Aliases

| Query Term | Automatically Expands To | Use Case |
|------------|--------------------------|----------|
| `tgeomanager` | `ggeomanager`, `gettopvolume`, `tgeonavigator`, `findnode` | Geometry management API variants |
| `navigation` | `findnode`, `getcurrentnode`, `tgeonavigator`, `ggeomanager` | Navigation/traversal methods |
| `open` | `tfile`, `tfile_open` | File opening operations |
| `storage` | `tclonesarray`, `pushtrack` | Data storage patterns |
| `assembly` | `addnode` | Geometry assembly methods |
| `processhits` | `override` | FairShip detector pattern |
| `shipfieldmaker` | `defineglobalfield`, `definelocalfield`, `defineregionfield` | FairShip field definitions |
| `setbranchaddress` | `getentry`, `ttree` | TTree branch address operations |

### Example: How Expansion Works

```bash
$ root-rag ask "TGeoManager navigation"

# Internal transformation:
# 1. Tokenize: ["tgeomanager", "navigation"]
# 2. Expand "navigation" → ["findnode", "getcurrentnode", "tgeonavigator", "ggeomanager"]
# 3. Final query: "ggeomanager getcurrentnode tgeonavigator findnode tgeomanager"

# Result: Finds chunks about geometry navigation, not just TGeoManager alone
```

## Practical Examples

### Finding ROOT API Definitions

```bash
# ✅ Find TTree methods
root-rag ask "TTree::Fill"
root-rag ask "TTree Branch"

# ✅ Find geometry classes
root-rag ask "TGeoManager MakeBox"
root-rag ask "TGeoVolume TGeoMedium"

# ✅ Find physics vectors
root-rag ask "TVector3 Magnitude"
root-rag ask "TLorentzVector Energy"
```

### Finding FairShip Patterns

```bash
# ✅ Find detector implementations
root-rag ask "ProcessHits DetectorHit"
root-rag ask "MuonShield geometry"

# ✅ Find data handling
root-rag ask "TTree Fill FairShip"
root-rag ask "TBranch SetAddress"
```

### Cross-Codebase Discovery

```bash
# ✅ See how FairShip uses ROOT
root-rag ask "TGeoManager AddNode FairShip"
root-rag ask "TTree Fill Branch"

# ✅ Find patterns in multiple places
root-rag ask "RDataFrame Filter"
root-rag ask "SOFIE ROperator"
```

## Debugging Failed Queries

### Problem: 0 Results

**Causes**:
1. Typo in keyword (most common)
2. Only stop words remain after filtering
3. Term not in indexed corpus

**Solutions**:
```bash
# Check your spelling
root-rag ask "TGeoManagr"      # ❌ No results (typo)
root-rag ask "TGeoManager"     # ✅ Results

# Use specific keywords
root-rag ask "I am searching for something"  # ❌ Only stop words
root-rag ask "TTree Fill"                    # ✅ Keywords

# Verify term is in corpus
root-rag versions              # See what's indexed
root-rag grep "ClassName"      # Test grep for quick lookup
```

### Problem: Too Many Results (>10)

**Causes**:
1. Query term too generic (common name)
2. Searching across all indices

**Solutions**:
```bash
# Be more specific
root-rag ask "Fill"            # ❌ ~50 results (too broad)
root-rag ask "TTree::Fill"     # ✅ ~5 results (specific)

# Limit results
root-rag ask "TTree" --top-k 3  # Get only top 3 results

# Search specific index
root-rag ask "ProcessHits" --root-ref v6-36-08  # ROOT only
```

### Problem: Wrong Results

**Causes**:
1. Query too generic (matches unrelated code)
2. Class not in corpus (different version)

**Solutions**:
```bash
# Check available versions
root-rag versions

# Search with more context
root-rag ask "RDataFrame Filter"     # Less generic than just "Filter"
root-rag ask "TGeoManager MakeBox"   # More specific than just "MakeBox"
```

## Command Reference

### `root-rag ask` - Question Answering

```bash
root-rag ask "QUERY" [OPTIONS]

Options:
  --root-ref TEXT          ROOT version (default: v6-36-08)
  --index-id TEXT          Explicit index ID
  --top-k INTEGER          Max results (default: 5)
```

**Tips**:
- Use keywords, not sentences
- Combine related terms for better results
- Use `--top-k` to see more results

### `root-rag grep` - Fast Keyword Search

```bash
root-rag grep "PATTERN" [OPTIONS]

Options:
  --root-ref TEXT          ROOT version (default: v6-36-08)
  --top-k INTEGER          Max results (default: 10)
```

**Tips**:
- Simpler interface than `ask`
- Good for quick lookups
- Use for testing queries before refining

### `root-rag versions` - List Indexed Versions

```bash
root-rag versions

# Output:
# ROOT Tier 1 (v6-36-08): 1,106 chunks, 53 files
# FairShip (master): 386 chunks, 163 files
# SOFIE (v6-36-08): 140 chunks, 40 files
```

## FAQ

**Q: Why does "Where is TGeoManager?" work but "Where is TGeoManager declared?" works better?**

A: Both filter to `["tgeomanager"]`, but:
- "Where is TGeoManager?" → FTS5 finds 5 results
- "Where is TGeoManager declared?" → FTS5 finds 1 result (the actual declaration)

The word "declared" is not filtered, so it narrows results to code that mentions both terms.

**Q: Can I search for operators like `::` or `->`?**

A: No, FTS5 tokenizes on word boundaries. Use the class/method name instead:
- ✅ `root-rag ask "TTree::Fill"` → searches for "TTree" and "Fill"
- ✅ `root-rag ask "ProcessHits"` → searches for "ProcessHits"

**Q: Does query order matter?**

A: No. `"TTree Fill"` and `"Fill TTree"` return the same results.

**Q: Why is my typo query returning 0 results?**

A: FTS5 requires exact word matches. Fix the typo:
- ❌ "searchin" (typo) → 0 results
- ✅ "searching" (correct) → results

**Q: Can I use wildcards or regex?**

A: Limited support. FTS5 accepts `*` for prefix matching:
```bash
root-rag ask "TGeo*"  # Matches TGeoManager, TGeoVolume, etc.
```

But regex patterns are not supported.

---

## Summary

| What | Query | Result |
|------|-------|--------|
| **Keywords work** | `TTree Fill` | ✅ Results |
| **Stop words filtered** | `Where is TTree?` | ✅ Works (becomes `TTree`) |
| **Order doesn't matter** | `Fill TTree` vs `TTree Fill` | ✅ Same results |
| **Typos fail** | `Treee Fill` | ❌ 0 results |
| **Specificity helps** | `TTree::Fill` | ✅ Better results |
| **Aliases expand** | `TGeoManager navigation` | ✅ Expanded search |

**Golden Rule**: Use **precise keywords** and **avoid natural language**. Think like you're searching code, not asking a chatbot.
