# Quick Query Reference

## TL;DR - Golden Rules

1. **Use keywords, not sentences**
   - ✅ `root-rag ask "TTree::Fill"`
   - ❌ `root-rag ask "How do I fill a tree?"`

2. **Exact spelling matters**
   - ✅ `root-rag ask "TGeoManager"` → results
   - ❌ `root-rag ask "TGeoManagr"` → 0 results

3. **More keywords = better results**
   - ✅ `root-rag ask "TTree Fill Branch"` → 5 results
   - ✅ `root-rag ask "TTree Fill"` → also works
   - ⚠️ `root-rag ask "Fill"` → too broad

4. **Stop words are ignored**
   - These words disappear: and, in, where, what, how, is, are, the, definition, usage, pattern, implementation, etc.
   - `"Where is TGeoManager?"` = `"TGeoManager"` (after filtering)

5. **Order doesn't matter**
   - `"TTree Fill"` = `"Fill TTree"` (same results)

## Common Query Patterns

### Finding ROOT API Definitions
```bash
root-rag ask "TTree::Fill"              # Method definition
root-rag ask "TGeoManager MakeBox"      # Geometry API
root-rag ask "TVector3 Magnitude"       # Physics vector methods
```

### Finding FairShip Code
```bash
root-rag ask "ProcessHits DetectorHit"  # Detector implementation
root-rag ask "TTree Fill FairShip"      # ROOT + FairShip usage
```

### Getting More Results
```bash
root-rag ask "TVector3" --top-k 10      # Get top 10 instead of 5
```

## Why Queries Fail

| Problem | Why | Solution |
|---------|-----|----------|
| 0 results | Typo or term not in index | Check spelling, use `root-rag versions` |
| Too many results | Query too generic | Add more keywords |
| Wrong results | Stop words removed meaning | Rephrase with more specific terms |
| Same results for different queries | Some words are stop words | Check [QUERY_SYNTAX_GUIDE.md](QUERY_SYNTAX_GUIDE.md) |

## Complete Stop Words List

Automatically removed from queries:
```
and, in, through, with, files, file, fairship, root, usage, pattern, 
implementation, implementations, overrides, override, modules, module, 
detectors, detector, definition, test, validation, global, local, object
```

## Full Documentation

📖 **Detailed guide:** [QUERY_SYNTAX_GUIDE.md](QUERY_SYNTAX_GUIDE.md)

This quick reference covers the essentials. For complete information including:
- How aliases work
- Debugging strategies
- Advanced query techniques
- FAQ

See the full guide.

## Examples That Work

```bash
# ROOT API searches
root-rag ask "TTree Branch"
root-rag ask "TH1F histogram"
root-rag ask "TVector3 dot product"

# FairShip patterns
root-rag ask "MuonShield detector"
root-rag ask "ProcessHits override"

# Cross-codebase
root-rag ask "TGeoManager AddNode FairShip"
root-rag ask "SOFIE ROperator Conv"
```

## Need Help?

1. **Query not working?** → Read [QUERY_SYNTAX_GUIDE.md - Debugging section](QUERY_SYNTAX_GUIDE.md#debugging-failed-queries)
2. **Want details?** → Read [QUERY_SYNTAX_GUIDE.md](QUERY_SYNTAX_GUIDE.md)
3. **Command help?** → Run `root-rag ask --help`
4. **Available indices?** → Run `root-rag versions`
