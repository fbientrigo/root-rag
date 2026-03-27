#!/usr/bin/env python3
"""
Extract ROOT usage evidence from FairShip source code.

This script scans a local FairShip clone to identify:
1. ROOT header includes
2. ROOT symbols (classes, namespaces)
3. Usage patterns by module

Outputs:
- JSON artifact: artifacts/fairship_root_usage_inventory.json
- Markdown report: reports/fairship_root_usage_inventory.md
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# File extensions to scan
SOURCE_EXTENSIONS = {'.cxx', '.cpp', '.cc', '.h', '.hpp', '.hh', '.C', '.py'}

# Directories to ignore
IGNORE_DIRS = {'.git', 'build', 'dist', '__pycache__', 'external', 'cmake-build'}


@dataclass
class UsageEntry:
    """Record of where a header/symbol appears."""
    name: str
    count: int
    files: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ModuleSummary:
    """Summary of ROOT usage in a module."""
    module_name: str
    files_count: int
    headers_count: int
    symbols_count: int
    top_headers: List[Tuple[str, int]]
    top_symbols: List[Tuple[str, int]]
    
    def to_dict(self):
        return {
            'module_name': self.module_name,
            'files_count': self.files_count,
            'headers_count': self.headers_count,
            'symbols_count': self.symbols_count,
            'top_headers': [{'name': h, 'count': c} for h, c in self.top_headers],
            'top_symbols': [{'name': s, 'count': c} for s, c in self.top_symbols],
        }


class FairShipROOTExtractor:
    """Extract ROOT usage from FairShip source code."""
    
    def __init__(self, fairship_path: Path):
        self.fairship_path = fairship_path
        self.scanned_files = 0
        self.matched_files = 0
        
        # Track headers: {header_name: {file_path, ...}}
        self.root_headers: Dict[str, Set[str]] = defaultdict(set)
        
        # Track symbols: {symbol: {file_path, ...}}
        self.root_symbols: Dict[str, Set[str]] = defaultdict(set)
        
        # Track by module: {module: {headers: {...}, symbols: {...}}}
        self.module_usage: Dict[str, Dict] = defaultdict(
            lambda: {'headers': defaultdict(int), 'symbols': defaultdict(int), 'files': set()}
        )
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for matching ROOT includes and symbols."""
        
        # ROOT header include patterns
        self.include_patterns = [
            # Direct ROOT headers with T prefix
            re.compile(r'#\s*include\s*[<"]([T][A-Za-z0-9_]+\.h(?:h|pp|xx)?)[">]'),
            # ROOT namespace paths
            re.compile(r'#\s*include\s*[<"](ROOT/[A-Za-z0-9_/]+\.h(?:h|pp|xx)?)[">]'),
            # TMVA headers
            re.compile(r'#\s*include\s*[<"](TMVA/[A-Za-z0-9_/]+\.h(?:h|pp|xx)?)[">]'),
            # Math headers
            re.compile(r'#\s*include\s*[<"](Math/[A-Za-z0-9_/]+\.h(?:h|pp|xx)?)[">]'),
            # Common ROOT headers
            re.compile(r'#\s*include\s*[<"](Rtypes\.h)[">]'),
            re.compile(r'#\s*include\s*[<"](RVersion\.h)[">]'),
            re.compile(r'#\s*include\s*[<"](Riostream\.h)[">]'),
        ]
        
        # ROOT symbol patterns (conservative to avoid false positives)
        self.symbol_patterns = [
            # T-prefixed classes (common ROOT pattern)
            re.compile(r'\b(T[A-Z][A-Za-z0-9_]*)\b'),
            # ROOT namespace
            re.compile(r'\b(ROOT::[A-Za-z0-9_:]+)\b'),
            # TMVA namespace
            re.compile(r'\b(TMVA::[A-Za-z0-9_:]+)\b'),
            # Math namespace
            re.compile(r'\b(Math::[A-Za-z0-9_:]+)\b'),
        ]
        
        # Common ROOT class patterns to prioritize
        self.common_root_classes = {
            'TObject', 'TClass', 'TNamed', 'TString',
            'TFile', 'TTree', 'TBranch', 'TLeaf',
            'TH1', 'TH1F', 'TH1D', 'TH2', 'TH2F', 'TH2D', 'TH3',
            'TGraph', 'TGraphErrors', 'TCanvas', 'TPad',
            'TVector2', 'TVector3', 'TLorentzVector',
            'TGeoManager', 'TGeoVolume', 'TGeoNode', 'TGeoMaterial',
            'TRandom', 'TRandom3',
            'TClonesArray', 'TObjArray', 'TList',
            'TF1', 'TF2', 'TF3',
            'TChain', 'TDirectory',
        }
    
    def _should_ignore_dir(self, dir_path: Path) -> bool:
        """Check if directory should be ignored."""
        parts = dir_path.parts
        for part in parts:
            if part.startswith('.'):
                return True
            if part in IGNORE_DIRS:
                return True
            if 'cmake-build' in part.lower():
                return True
        return False
    
    def _get_module_name(self, file_path: Path) -> str:
        """Extract top-level module/directory name."""
        try:
            rel_path = file_path.relative_to(self.fairship_path)
            if len(rel_path.parts) > 0:
                return rel_path.parts[0]
        except ValueError:
            pass
        return 'unknown'
    
    def _extract_includes(self, content: str, file_path: str, module: str):
        """Extract ROOT header includes from file content."""
        for pattern in self.include_patterns:
            for match in pattern.finditer(content):
                header = match.group(1)
                self.root_headers[header].add(file_path)
                self.module_usage[module]['headers'][header] += 1
    
    def _extract_symbols(self, content: str, file_path: str, module: str):
        """Extract ROOT symbols from file content."""
        found_symbols = set()
        
        for pattern in self.symbol_patterns:
            for match in pattern.finditer(content):
                symbol = match.group(1)
                # Filter out some obvious false positives
                if self._is_likely_root_symbol(symbol):
                    found_symbols.add(symbol)
        
        # Add to tracking
        for symbol in found_symbols:
            self.root_symbols[symbol].add(file_path)
            self.module_usage[module]['symbols'][symbol] += 1
    
    def _is_likely_root_symbol(self, symbol: str) -> bool:
        """Filter to keep only likely ROOT symbols."""
        # Known ROOT classes
        if symbol in self.common_root_classes:
            return True
        
        # ROOT, TMVA, Math namespaces
        if symbol.startswith(('ROOT::', 'TMVA::', 'Math::')):
            return True
        
        # T-prefixed classes that look like ROOT
        if symbol.startswith('T'):
            # Common ROOT prefixes
            if symbol.startswith(('TGeo', 'TMVA', 'TTree', 'TH1', 'TH2', 'TH3', 'TGraph')):
                return True
            # Must be at least 3 chars and start with T + uppercase
            if len(symbol) >= 3 and symbol[1].isupper():
                return True
        
        return False
    
    def scan_file(self, file_path: Path):
        """Scan a single file for ROOT usage."""
        self.scanned_files += 1
        module = self._get_module_name(file_path)
        
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = file_path.read_text(encoding='latin-1')
            
            # Track whether this file has any ROOT usage
            initial_headers = sum(len(files) for files in self.root_headers.values())
            initial_symbols = sum(len(files) for files in self.root_symbols.values())
            
            # Extract includes and symbols
            file_str = str(file_path.relative_to(self.fairship_path))
            self._extract_includes(content, file_str, module)
            self._extract_symbols(content, file_str, module)
            
            # Check if this file had any ROOT usage
            final_headers = sum(len(files) for files in self.root_headers.values())
            final_symbols = sum(len(files) for files in self.root_symbols.values())
            
            if final_headers > initial_headers or final_symbols > initial_symbols:
                self.matched_files += 1
                self.module_usage[module]['files'].add(file_str)
                
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
    
    def scan_directory(self):
        """Recursively scan FairShip directory."""
        print(f"Scanning FairShip at: {self.fairship_path}")
        
        for file_path in self.fairship_path.rglob('*'):
            # Skip if in ignored directory
            if self._should_ignore_dir(file_path):
                continue
            
            # Check if it's a source file
            if file_path.is_file() and file_path.suffix in SOURCE_EXTENSIONS:
                self.scan_file(file_path)
        
        print(f"Scanned {self.scanned_files} files, found ROOT usage in {self.matched_files} files")
    
    def generate_json_report(self, output_path: Path):
        """Generate machine-readable JSON report."""
        
        # Convert headers to sorted list
        headers_list = [
            {
                'name': header,
                'count': len(files),
                'files': sorted(list(files))[:50]  # Limit to 50 files per header
            }
            for header, files in sorted(
                self.root_headers.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
        ]
        
        # Convert symbols to sorted list
        symbols_list = [
            {
                'name': symbol,
                'count': len(files),
                'files': sorted(list(files))[:50]  # Limit to 50 files per symbol
            }
            for symbol, files in sorted(
                self.root_symbols.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
        ]
        
        # Generate per-module summary
        module_summaries = []
        for module, usage in sorted(self.module_usage.items()):
            headers = usage['headers']
            symbols = usage['symbols']
            files = usage['files']
            
            top_headers = sorted(headers.items(), key=lambda x: x[1], reverse=True)[:10]
            top_symbols = sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]
            
            module_summaries.append({
                'module_name': module,
                'files_count': len(files),
                'headers_count': len(headers),
                'symbols_count': len(symbols),
                'top_headers': [{'name': h, 'count': c} for h, c in top_headers],
                'top_symbols': [{'name': s, 'count': c} for s, c in top_symbols],
            })
        
        report = {
            'scanned_files_count': self.scanned_files,
            'matched_files_count': self.matched_files,
            'total_root_headers': len(self.root_headers),
            'total_root_symbols': len(self.root_symbols),
            'root_headers': headers_list,
            'root_symbols': symbols_list,
            'per_module_summary': module_summaries,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report written to: {output_path}")
    
    def generate_markdown_report(self, output_path: Path):
        """Generate human-readable Markdown report."""
        
        lines = []
        lines.append("# FairShip ROOT Usage Inventory")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Scanned files**: {self.scanned_files:,}")
        lines.append(f"- **Files with ROOT usage**: {self.matched_files:,}")
        lines.append(f"- **Unique ROOT headers**: {len(self.root_headers):,}")
        lines.append(f"- **Unique ROOT symbols**: {len(self.root_symbols):,}")
        lines.append("")
        
        # Top 20 ROOT headers
        lines.append("## Top 20 ROOT Headers")
        lines.append("")
        lines.append("| Rank | Header | Usage Count | Sample Files |")
        lines.append("|------|--------|-------------|--------------|")
        
        sorted_headers = sorted(
            self.root_headers.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]
        
        for idx, (header, files) in enumerate(sorted_headers, 1):
            sample_files = ', '.join(list(files)[:3])
            if len(files) > 3:
                sample_files += f", ... ({len(files) - 3} more)"
            lines.append(f"| {idx} | `{header}` | {len(files)} | {sample_files} |")
        
        lines.append("")
        
        # Top 20 ROOT symbols
        lines.append("## Top 20 ROOT Symbols")
        lines.append("")
        lines.append("| Rank | Symbol | Usage Count | Sample Files |")
        lines.append("|------|--------|-------------|--------------|")
        
        sorted_symbols = sorted(
            self.root_symbols.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]
        
        for idx, (symbol, files) in enumerate(sorted_symbols, 1):
            sample_files = ', '.join(list(files)[:3])
            if len(files) > 3:
                sample_files += f", ... ({len(files) - 3} more)"
            lines.append(f"| {idx} | `{symbol}` | {len(files)} | {sample_files} |")
        
        lines.append("")
        
        # Usage by module
        lines.append("## ROOT Usage by Module")
        lines.append("")
        lines.append("| Module | Files | Headers | Symbols | Top Header | Top Symbol |")
        lines.append("|--------|-------|---------|---------|------------|------------|")
        
        for module, usage in sorted(self.module_usage.items(), key=lambda x: len(x[1]['files']), reverse=True):
            files_count = len(usage['files'])
            headers_count = len(usage['headers'])
            symbols_count = len(usage['symbols'])
            
            top_header = ''
            if usage['headers']:
                top_h = max(usage['headers'].items(), key=lambda x: x[1])
                top_header = f"`{top_h[0]}` ({top_h[1]})"
            
            top_symbol = ''
            if usage['symbols']:
                top_s = max(usage['symbols'].items(), key=lambda x: x[1])
                top_symbol = f"`{top_s[0]}` ({top_s[1]})"
            
            lines.append(f"| {module} | {files_count} | {headers_count} | {symbols_count} | {top_header} | {top_symbol} |")
        
        lines.append("")
        
        # Limitations
        lines.append("## Methodology & Limitations")
        lines.append("")
        lines.append("### Extraction Method")
        lines.append("- **Approach**: Regex-based pattern matching")
        lines.append("- **File types**: .cxx, .cpp, .cc, .h, .hpp, .hh, .C, .py")
        lines.append("- **Ignored directories**: .git, build, cmake-build*, dist, __pycache__, external")
        lines.append("")
        lines.append("### Known Limitations")
        lines.append("1. **No AST parsing**: This is a text-based analysis, not semantic parsing")
        lines.append("2. **False positives**: Some T-prefixed symbols may not be ROOT classes")
        lines.append("3. **False negatives**: May miss ROOT usage in preprocessor macros or generated code")
        lines.append("4. **Symbol counts**: Counts file occurrences, not actual usage frequency within files")
        lines.append("")
        lines.append("### Confidence Level")
        lines.append("- **Headers**: HIGH (direct #include statements)")
        lines.append("- **Common symbols** (TFile, TTree, TH1, etc.): HIGH")
        lines.append("- **Generic T-prefixed symbols**: MEDIUM (some may be non-ROOT)")
        lines.append("")
        
        # Next steps
        lines.append("## Suggested Next Steps")
        lines.append("")
        lines.append("1. **Derive Tier 1/Tier 2 corpus**:")
        lines.append("   - Tier 1: Top 20-30 most-used headers/symbols")
        lines.append("   - Tier 2: Modules/subsystems heavily used (e.g., TGeo*, TMVA)")
        lines.append("")
        lines.append("2. **Validate findings**:")
        lines.append("   - Cross-reference with ROOT documentation")
        lines.append("   - Manual spot-checking of high-frequency symbols")
        lines.append("")
        lines.append("3. **Update retrieval ranking**:")
        lines.append("   - Boost Tier 1 content in vector store")
        lines.append("   - Ensure Tier 1 documentation is indexed")
        lines.append("")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Markdown report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ROOT usage evidence from FairShip source code"
    )
    parser.add_argument(
        '--fairship-path',
        type=Path,
        required=True,
        help='Path to local FairShip clone (e.g., ../FairShip)'
    )
    parser.add_argument(
        '--json-output',
        type=Path,
        default=Path('artifacts/fairship_root_usage_inventory.json'),
        help='Output path for JSON artifact'
    )
    parser.add_argument(
        '--markdown-output',
        type=Path,
        default=Path('reports/fairship_root_usage_inventory.md'),
        help='Output path for Markdown report'
    )
    
    args = parser.parse_args()
    
    # Validate FairShip path
    if not args.fairship_path.exists():
        print(f"Error: FairShip path does not exist: {args.fairship_path}")
        return 1
    
    if not args.fairship_path.is_dir():
        print(f"Error: FairShip path is not a directory: {args.fairship_path}")
        return 1
    
    # Run extraction
    extractor = FairShipROOTExtractor(args.fairship_path)
    extractor.scan_directory()
    
    # Generate reports
    extractor.generate_json_report(args.json_output)
    extractor.generate_markdown_report(args.markdown_output)
    
    print("\nExtraction complete!")
    print(f"  JSON: {args.json_output}")
    print(f"  Markdown: {args.markdown_output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
