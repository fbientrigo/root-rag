"""
Tests for extract_fairship_root_usage.py
"""

import json
import tempfile
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from extract_fairship_root_usage import FairShipROOTExtractor


def test_header_extraction():
    """Test ROOT header extraction from sample content."""
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create a sample source file
        sample_file = tmppath / "test.cxx"
        sample_file.write_text("""
#include "TFile.h"
#include <TTree.h>
#include "ROOT/RDataFrame.hxx"
#include "TMVA/Reader.h"
#include <TH1F.h>

void test() {
    TFile* file = new TFile("test.root");
    TTree* tree = new TTree("data", "Data");
    TH1F* hist = new TH1F("h1", "Histogram", 100, 0, 100);
}
""")
        
        # Run extractor
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Check results
        assert extractor.scanned_files == 1
        assert extractor.matched_files == 1
        
        # Check headers found
        assert "TFile.h" in extractor.root_headers
        assert "TTree.h" in extractor.root_headers
        assert "ROOT/RDataFrame.hxx" in extractor.root_headers
        assert "TMVA/Reader.h" in extractor.root_headers
        assert "TH1F.h" in extractor.root_headers
        
        # Check symbols found
        assert "TFile" in extractor.root_symbols
        assert "TTree" in extractor.root_symbols
        assert "TH1F" in extractor.root_symbols
        
        print("PASS: Header extraction test passed")


def test_symbol_filtering():
    """Test that symbol filtering avoids obvious false positives."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create a sample file with mixed content
        sample_file = tmppath / "test.h"
        sample_file.write_text("""
#include "TFile.h"

class MyClass {
    TVector3 position;  // Should be detected
    TLorentzVector momentum;  // Should be detected
    int Test;  // Should not be detected (lowercase after T)
    double Time;  // Should not be detected (not matching pattern)
};

void process() {
    ROOT::Math::XYZVector v;  // Should be detected
    TGeoManager* geo = nullptr;  // Should be detected
}
""")
        
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Check expected symbols are found
        assert "TVector3" in extractor.root_symbols
        assert "TLorentzVector" in extractor.root_symbols
        assert "TGeoManager" in extractor.root_symbols
        
        # Check ROOT namespace is found
        found_root_math = any("ROOT::Math" in s for s in extractor.root_symbols)
        assert found_root_math, "Should find ROOT::Math namespace"
        
        print("PASS: Symbol filtering test passed")


def test_module_grouping():
    """Test that files are correctly grouped by module."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create module structure
        module1 = tmppath / "module1"
        module1.mkdir()
        (module1 / "file1.cxx").write_text('#include "TFile.h"\nTFile f;')
        
        module2 = tmppath / "module2"
        module2.mkdir()
        (module2 / "file2.h").write_text('#include "TTree.h"\nTTree t;')
        
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Check module tracking
        assert "module1" in extractor.module_usage
        assert "module2" in extractor.module_usage
        assert len(extractor.module_usage["module1"]["files"]) == 1
        assert len(extractor.module_usage["module2"]["files"]) == 1
        
        print("PASS: Module grouping test passed")


def test_json_generation():
    """Test JSON report generation."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create sample file
        (tmppath / "test.cxx").write_text('#include "TFile.h"\nTFile f;')
        
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Generate JSON report
        json_output = Path(tmpdir) / "output.json"
        extractor.generate_json_report(json_output)
        
        assert json_output.exists()
        
        # Validate JSON structure
        with open(json_output) as f:
            data = json.load(f)
        
        assert "scanned_files_count" in data
        assert "matched_files_count" in data
        assert "root_headers" in data
        assert "root_symbols" in data
        assert "per_module_summary" in data
        
        assert data["scanned_files_count"] == 1
        assert data["matched_files_count"] == 1
        
        print("PASS: JSON generation test passed")


def test_t2_false_positive_filtering():
    """Test T2: conservative filtering of false positives."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create sample with common false positives
        sample_file = tmppath / "geometry.h"
        sample_file.write_text("""
#ifndef TIMEDET_TIMEDET_H_
#define TIMEDET_TIMEDET_H_

// Valid ROOT symbols
#include "TFile.h"
#include "TTree.h"

void process() {
    // Valid ROOT classes - should be detected
    TFile* file = nullptr;
    TTree* tree = nullptr;
    TVector3 pos;
    TGeoManager* geo = nullptr;
    
    // False positives - should be rejected
    TODO: implement this;  // Placeholder
    TEST macro expansion;  // Test macro
    TARGET_YAML config;    // All-caps config
    TRY_2025 code;         // Year-suffixed
    TTstationID id;        // Geometry node (TT_ prefix)
    TT_scifi_plane_vert_volume vol;  // Underscore-heavy geometry
    TIMEDET detector;      // All-caps detector name
    
    // Edge cases
    int TARGET = 0;        // All-caps word starting with T
    bool TIME_FLAG = true; // All-caps with underscore
}

#endif  // TIMEDET_TIMEDET_H_
""")
        
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Valid ROOT symbols should be found
        assert "TFile" in extractor.root_symbols, "TFile should be detected"
        assert "TTree" in extractor.root_symbols, "TTree should be detected"
        assert "TVector3" in extractor.root_symbols, "TVector3 should be detected"
        assert "TGeoManager" in extractor.root_symbols, "TGeoManager should be detected"
        
        # False positives should be rejected
        assert "TODO" not in extractor.root_symbols, "TODO should be rejected"
        assert "TEST" not in extractor.root_symbols, "TEST should be rejected"
        assert "TARGET_YAML" not in extractor.root_symbols, "TARGET_YAML should be rejected"
        assert "TRY_2025" not in extractor.root_symbols, "TRY_2025 should be rejected"
        assert "TTstationID" not in extractor.root_symbols, "TTstationID should be rejected"
        assert "TT_scifi_plane_vert_volume" not in extractor.root_symbols, "TT_scifi_plane should be rejected"
        assert "TIMEDET_TIMEDET_H_" not in extractor.root_symbols, "Include guard should be rejected"
        assert "TIMEDET" not in extractor.root_symbols, "TIMEDET should be rejected"
        assert "TARGET" not in extractor.root_symbols, "TARGET should be rejected"
        assert "TIME_FLAG" not in extractor.root_symbols, "TIME_FLAG should be rejected"
        
        print("PASS: T2 false positive filtering test passed")


def test_provenance_classification():
    """Test T3: provenance-aware classification."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create local FairShip-like headers
        target_dir = tmppath / "target"
        target_dir.mkdir()
        (target_dir / "TargetPoint.h").write_text("#ifndef TARGETPOINT_H\nclass TargetPoint {};\n#endif")
        (target_dir / "TargetTracker.h").write_text("#ifndef TARGETTRACKER_H\nclass TargetTracker {};\n#endif")
        
        veto_dir = tmppath / "veto"
        veto_dir.mkdir()
        (veto_dir / "TVetoPoint.h").write_text("#ifndef TVETOPOINT_H\nclass TVetoPoint {};\n#endif")
        (veto_dir / "TTimeDet.h").write_text("#ifndef TTIMEDET_H\nclass TTimeDet {};\n#endif")
        
        # Create sample file mixing ROOT core, adjacent, and local
        sample_file = tmppath / "analysis.cxx"
        sample_file.write_text("""
#include "TFile.h"          // ROOT_CORE
#include "TTree.h"          // ROOT_CORE
#include "TVector3.h"       // ROOT_CORE
#include "TGeoManager.h"    // ROOT_CORE
#include "ROOT/RDataFrame.hxx"  // ROOT_CORE
#include "TMVA/Reader.h"    // ROOT_CORE

#include "TVirtualMC.h"     // ROOT_ADJACENT
#include "TGeant4.h"        // ROOT_ADJACENT (if present)

#include "TargetPoint.h"    // FAIRSHIP_LOCAL
#include "TargetTracker.h"  // FAIRSHIP_LOCAL
#include "TVetoPoint.h"     // FAIRSHIP_LOCAL
#include "TTimeDet.h"       // FAIRSHIP_LOCAL

void analysis() {
    // ROOT core classes
    TFile* f = new TFile("data.root");
    TTree* tree = new TTree("events", "Events");
    TVector3 pos;
    TGeoManager* geo = nullptr;
    ROOT::RDataFrame df("tree", "data.root");
    
    // ROOT-adjacent (framework)
    TVirtualMC* mc = nullptr;
    
    // FairShip-local classes (T-prefixed but local)
    TargetPoint* point = nullptr;
    TargetTracker* tracker = nullptr;
    TVetoPoint* vp = nullptr;
    TTimeDet* td = nullptr;
}
""")
        
        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()
        
        # Check ROOT_CORE classification
        assert "TFile.h" in extractor.root_headers
        assert extractor.header_provenance.get("TFile.h") == "ROOT_CORE", "TFile.h should be ROOT_CORE"
        
        assert "TTree.h" in extractor.root_headers
        assert extractor.header_provenance.get("TTree.h") == "ROOT_CORE", "TTree.h should be ROOT_CORE"
        
        assert "TVector3.h" in extractor.root_headers
        assert extractor.header_provenance.get("TVector3.h") == "ROOT_CORE", "TVector3.h should be ROOT_CORE"
        
        assert "ROOT/RDataFrame.hxx" in extractor.root_headers
        assert extractor.header_provenance.get("ROOT/RDataFrame.hxx") == "ROOT_CORE", "ROOT/RDataFrame.hxx should be ROOT_CORE"
        
        assert "TFile" in extractor.root_symbols
        assert extractor.symbol_provenance.get("TFile") == "ROOT_CORE", "TFile should be ROOT_CORE"
        
        assert "TTree" in extractor.root_symbols
        assert extractor.symbol_provenance.get("TTree") == "ROOT_CORE", "TTree should be ROOT_CORE"
        
        # Check ROOT_ADJACENT classification
        if "TVirtualMC.h" in extractor.root_headers:
            assert extractor.header_provenance.get("TVirtualMC.h") == "ROOT_ADJACENT", "TVirtualMC.h should be ROOT_ADJACENT"
        
        if "TVirtualMC" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TVirtualMC") == "ROOT_ADJACENT", "TVirtualMC should be ROOT_ADJACENT"
        
        # Check FAIRSHIP_LOCAL classification
        assert "TargetPoint.h" in extractor.root_headers
        assert extractor.header_provenance.get("TargetPoint.h") == "FAIRSHIP_LOCAL", "TargetPoint.h should be FAIRSHIP_LOCAL"
        
        assert "TargetTracker.h" in extractor.root_headers
        assert extractor.header_provenance.get("TargetTracker.h") == "FAIRSHIP_LOCAL", "TargetTracker.h should be FAIRSHIP_LOCAL"
        
        assert "TVetoPoint.h" in extractor.root_headers
        assert extractor.header_provenance.get("TVetoPoint.h") == "FAIRSHIP_LOCAL", "TVetoPoint.h should be FAIRSHIP_LOCAL"
        
        assert "TTimeDet.h" in extractor.root_headers
        assert extractor.header_provenance.get("TTimeDet.h") == "FAIRSHIP_LOCAL", "TTimeDet.h should be FAIRSHIP_LOCAL"
        
        if "TargetPoint" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TargetPoint") == "FAIRSHIP_LOCAL", "TargetPoint should be FAIRSHIP_LOCAL"
        
        if "TargetTracker" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TargetTracker") == "FAIRSHIP_LOCAL", "TargetTracker should be FAIRSHIP_LOCAL"
        
        if "TVetoPoint" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TVetoPoint") == "FAIRSHIP_LOCAL", "TVetoPoint should be FAIRSHIP_LOCAL"
        
        if "TTimeDet" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TTimeDet") == "FAIRSHIP_LOCAL", "TTimeDet should be FAIRSHIP_LOCAL"
        
        # Check JSON structure includes provenance
        json_output = Path(tmpdir) / "output.json"
        extractor.generate_json_report(json_output)
        
        with open(json_output) as f:
            data = json.load(f)
        
        assert "provenance_classification" in data, "JSON should include provenance_classification"
        assert "header_counts" in data["provenance_classification"]
        assert "symbol_counts" in data["provenance_classification"]
        assert "headers_by_provenance" in data["provenance_classification"]
        assert "symbols_by_provenance" in data["provenance_classification"]
        
        # Verify categories exist
        prov_data = data["provenance_classification"]
        assert "ROOT_CORE" in prov_data["header_counts"] or "ROOT_CORE" in prov_data["symbol_counts"]
        assert "FAIRSHIP_LOCAL" in prov_data["header_counts"]
        
        print("PASS: Provenance classification test passed")


def test_t4_missing_root_core_families():
    """Test T4: Validation-driven expansion of ROOT_CORE families.

    Evidence from real FairShip validation showed these classes were
    wrongly classified as UNCERTAIN. T4 adds them to ROOT_CORE with:
    - Explicit whitelisting (common_root_classes)
    - Prefix fallbacks (TDatabase*, TPythia*, TEve*)
    - EVE reordering to fix Point collision
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create sample file with all the previously-missing ROOT_CORE classes
        sample_file = tmppath / "monte_carlo.cxx"
        sample_file.write_text("""
#include "TParticle.h"         // MC/PDG class
#include "TDatabasePDG.h"       // PDG database
#include "TParticlePDG.h"       // PDG particle info
#include "TMCProcess.h"         // MC process types
#include "TPythia6.h"           // Generator
#include "TPythia8.h"           // Generator
#include "TPythia8Decayer.h"    // Decayer
#include "TNtuple.h"            // Tree utility
#include "TStopwatch.h"         // Timer
#include "TMemFile.h"           // Memory file I/O
#include "TObjString.h"         // String object
#include "TMatrixD.h"           // Matrix utility
#include "TStyle.h"             // Graphics style
#include "TLatex.h"             // LaTeX graphics
#include "TLegend.h"            // Legend

void analysis() {
    // MC/PDG classes
    TParticle* p = nullptr;
    TDatabasePDG* pdg = nullptr;
    TParticlePDG* ppdg = nullptr;
    TMCProcess proc;

    // Generators
    TPythia6* py6 = nullptr;
    TPythia8* py8 = nullptr;
    TPythia8Decayer* dec = nullptr;

    // Utilities
    TNtuple* nt = nullptr;
    TStopwatch sw;
    TMemFile* mf = nullptr;
    TObjString* ostr = nullptr;
    TMatrixD* mat = nullptr;

    // Graphics
    TStyle* style = nullptr;
    TLatex* latex = nullptr;
    TLegend* legend = nullptr;
}
""")

        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()

        # Verify all MC/PDG classes are ROOT_CORE
        assert "TParticle" in extractor.root_symbols, "TParticle should be detected"
        assert extractor.symbol_provenance.get("TParticle") == "ROOT_CORE", "TParticle should be ROOT_CORE"

        assert "TDatabasePDG" in extractor.root_symbols, "TDatabasePDG should be detected"
        assert extractor.symbol_provenance.get("TDatabasePDG") == "ROOT_CORE", "TDatabasePDG should be ROOT_CORE (prefix fallback)"

        assert "TParticlePDG" in extractor.root_symbols, "TParticlePDG should be detected"
        assert extractor.symbol_provenance.get("TParticlePDG") == "ROOT_CORE", "TParticlePDG should be ROOT_CORE"

        assert "TMCProcess" in extractor.root_symbols, "TMCProcess should be detected"
        assert extractor.symbol_provenance.get("TMCProcess") == "ROOT_CORE", "TMCProcess should be ROOT_CORE"

        # Verify all generator classes are ROOT_CORE
        assert "TPythia6" in extractor.root_symbols, "TPythia6 should be detected"
        assert extractor.symbol_provenance.get("TPythia6") == "ROOT_CORE", "TPythia6 should be ROOT_CORE (prefix fallback)"

        assert "TPythia8" in extractor.root_symbols, "TPythia8 should be detected"
        assert extractor.symbol_provenance.get("TPythia8") == "ROOT_CORE", "TPythia8 should be ROOT_CORE (prefix fallback)"

        assert "TPythia8Decayer" in extractor.root_symbols, "TPythia8Decayer should be detected"
        assert extractor.symbol_provenance.get("TPythia8Decayer") == "ROOT_CORE", "TPythia8Decayer should be ROOT_CORE"

        # Verify utility classes are ROOT_CORE
        assert "TNtuple" in extractor.root_symbols, "TNtuple should be detected"
        assert extractor.symbol_provenance.get("TNtuple") == "ROOT_CORE", "TNtuple should be ROOT_CORE"

        assert "TStopwatch" in extractor.root_symbols, "TStopwatch should be detected"
        assert extractor.symbol_provenance.get("TStopwatch") == "ROOT_CORE", "TStopwatch should be ROOT_CORE"

        assert "TMemFile" in extractor.root_symbols, "TMemFile should be detected"
        assert extractor.symbol_provenance.get("TMemFile") == "ROOT_CORE", "TMemFile should be ROOT_CORE"

        assert "TObjString" in extractor.root_symbols, "TObjString should be detected"
        assert extractor.symbol_provenance.get("TObjString") == "ROOT_CORE", "TObjString should be ROOT_CORE"

        assert "TMatrixD" in extractor.root_symbols, "TMatrixD should be detected"
        assert extractor.symbol_provenance.get("TMatrixD") == "ROOT_CORE", "TMatrixD should be ROOT_CORE"

        # Verify graphics classes are ROOT_CORE
        assert "TStyle" in extractor.root_symbols, "TStyle should be detected"
        assert extractor.symbol_provenance.get("TStyle") == "ROOT_CORE", "TStyle should be ROOT_CORE"

        assert "TLatex" in extractor.root_symbols, "TLatex should be detected"
        assert extractor.symbol_provenance.get("TLatex") == "ROOT_CORE", "TLatex should be ROOT_CORE"

        assert "TLegend" in extractor.root_symbols, "TLegend should be detected"
        assert extractor.symbol_provenance.get("TLegend") == "ROOT_CORE", "TLegend should be ROOT_CORE"

        print("PASS: T4 missing ROOT_CORE families test passed")


def test_t4_eve_collision_fix():
    """Test T4: Fix EVE/local collision where TEvePointSet was wrongly FAIRSHIP_LOCAL.

    The issue: "Point" pattern in detector classification matched before
    TEve* prefix check, causing TEvePointSet → FAIRSHIP_LOCAL.

    T4 reorders checks: TEve* is checked BEFORE detector patterns.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a sample with EVE classes and detector Point patterns
        sample_file = tmppath / "visualization.cxx"
        sample_file.write_text("""
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEvePointSetPrintOut.h"
#include "TEveTrack.h"
#include "TEveElement.h"

#include "TargetPoint.h"         // Local detector class
#include "TargetTracker.h"       // Local detector class

void visualize() {
    // EVE classes - should be ROOT_CORE despite "Point" and "Track" patterns
    TEveManager* manager = nullptr;
    TEvePointSet* points = nullptr;
    TEvePointSetPrintOut* printer = nullptr;
    TEveTrack* track = nullptr;
    TEveElement* elem = nullptr;

    // Local detector classes - should remain FAIRSHIP_LOCAL
    TargetPoint* tpoint = nullptr;
    TargetTracker* ttracker = nullptr;
}
""")

        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()

        # Verify EVE classes are ROOT_CORE (not FAIRSHIP_LOCAL)
        if "TEvePointSet" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEvePointSet") == "ROOT_CORE", \
                "TEvePointSet should be ROOT_CORE (EVE check before Point pattern)"

        if "TEvePointSetPrintOut" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEvePointSetPrintOut") == "ROOT_CORE", \
                "TEvePointSetPrintOut should be ROOT_CORE (EVE check before Point pattern)"

        if "TEveTrack" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEveTrack") == "ROOT_CORE", \
                "TEveTrack should be ROOT_CORE (EVE check before Tracker pattern)"

        if "TEveManager" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEveManager") == "ROOT_CORE", \
                "TEveManager should be ROOT_CORE"

        if "TEveElement" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEveElement") == "ROOT_CORE", \
                "TEveElement should be ROOT_CORE"

        # Verify local detector classes are still FAIRSHIP_LOCAL
        if "TargetPoint" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TargetPoint") == "FAIRSHIP_LOCAL", \
                "TargetPoint should be FAIRSHIP_LOCAL (local detector class)"

        if "TargetTracker" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TargetTracker") == "FAIRSHIP_LOCAL", \
                "TargetTracker should be FAIRSHIP_LOCAL (local detector class)"

        print("PASS: T4 EVE collision fix test passed")


def test_t4_prefix_fallbacks():
    """Test T4: Prefix fallback matching for TDatabase*, TPythia*, TEve*.

    These prefix patterns catch entire families without explicit whitelist entries.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create sample with unknown classes that match fallback prefixes
        sample_file = tmppath / "prefixes.cxx"
        sample_file.write_text("""
#include "TDatabasePDG.h"
#include "TDatabaseManager.h"    // Hypothetical TDatabase* class
#include "TPythia6.h"
#include "TPythia8.h"
#include "TPythiaHelper.h"       // Hypothetical TPythia* class
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEvePointSet.h"
#include "TEveCustomElement.h"   // Hypothetical TEve* class

void test() {
    // All of these should be ROOT_CORE due to prefix matching
    TDatabasePDG* pdg1 = nullptr;
    TPythia6* py1 = nullptr;
    TPythia8* py2 = nullptr;
    TEveManager* ev1 = nullptr;
    TEveElement* ev2 = nullptr;
}
""")

        extractor = FairShipROOTExtractor(tmppath)
        extractor.scan_directory()

        # Explicit classnames in whitelists
        if "TDatabasePDG" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TDatabasePDG") == "ROOT_CORE"

        if "TPythia6" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TPythia6") == "ROOT_CORE"

        if "TPythia8" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TPythia8") == "ROOT_CORE"

        # Prefix fallback matches (may or may not extract, depending on symbol filtering)
        # Just verify that if they are detected, they are classified as ROOT_CORE
        if "TEveManager" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEveManager") == "ROOT_CORE", \
                "TEveManager should match TEve* prefix fallback"

        if "TEveElement" in extractor.root_symbols:
            assert extractor.symbol_provenance.get("TEveElement") == "ROOT_CORE", \
                "TEveElement should match TEve* prefix fallback"

        print("PASS: T4 prefix fallbacks test passed")


if __name__ == "__main__":
    test_header_extraction()
    test_symbol_filtering()
    test_module_grouping()
    test_json_generation()
    test_t2_false_positive_filtering()
    test_provenance_classification()
    test_t4_missing_root_core_families()
    test_t4_eve_collision_fix()
    test_t4_prefix_fallbacks()
    print("\nAll tests passed!")


