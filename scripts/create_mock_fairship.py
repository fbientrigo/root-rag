#!/usr/bin/env python3
"""
Create a minimal mock FairShip structure for testing the extraction script.

This creates a small sample directory tree with typical FairShip-like ROOT usage
so you can test the extraction script without needing the full FairShip clone.
"""

import tempfile
from pathlib import Path


def create_mock_fairship():
    """Create a mock FairShip directory structure."""
    
    tmpdir = Path(tempfile.mkdtemp(prefix="mock_fairship_"))
    print(f"Creating mock FairShip at: {tmpdir}")
    
    # Create module structure
    modules = {
        'veto': [
            'vetoPoint.cxx',
            'vetoHit.h',
        ],
        'ecal': [
            'ecalCell.cxx',
            'ecalReconstruction.h',
        ],
        'python': [
            'analysis.py',
        ],
    }
    
    for module, files in modules.items():
        module_dir = tmpdir / module
        module_dir.mkdir()
        
        for filename in files:
            file_path = module_dir / filename
            
            # Generate appropriate content
            if filename.endswith('.cxx'):
                content = f'''
#include "{filename.replace('.cxx', '.h')}"
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"
#include "TGeoManager.h"
#include <TClonesArray.h>

void {module}_process() {{
    TFile* outputFile = new TFile("output.root", "RECREATE");
    TTree* tree = new TTree("data", "Processed data");
    
    TVector3 position;
    TLorentzVector momentum;
    TClonesArray* hits = new TClonesArray("Hit", 100);
    
    TGeoManager* geoMan = TGeoManager::Import("geometry.root");
    
    // Process events
    for (int i = 0; i < 1000; i++) {{
        position.SetXYZ(i * 0.1, i * 0.2, i * 0.3);
        tree->Fill();
    }}
    
    outputFile->Write();
    outputFile->Close();
}}
'''
            elif filename.endswith('.h'):
                content = f'''
#ifndef {module.upper()}_H
#define {module.upper()}_H

#include "TObject.h"
#include "TVector3.h"
#include <ROOT/RDataFrame.hxx>

class {module.capitalize()}Point : public TObject {{
private:
    TVector3 fPosition;
    Double_t fEnergy;
    
public:
    {module.capitalize()}Point();
    virtual ~{module.capitalize()}Point();
    
    void SetPosition(const TVector3& pos) {{ fPosition = pos; }}
    TVector3 GetPosition() const {{ return fPosition; }}
    
    void SetEnergy(Double_t e) {{ fEnergy = e; }}
    Double_t GetEnergy() const {{ return fEnergy; }}
    
    ClassDef({module.capitalize()}Point, 1)
}};

#endif
'''
            elif filename.endswith('.py'):
                content = f'''
import ROOT
from ROOT import TFile, TTree, TH1F, TCanvas

def analyze_{module}():
    """Analyze {module} data using ROOT."""
    
    # Open ROOT file
    f = TFile.Open("data.root")
    tree = f.Get("data")
    
    # Create histogram
    hist = TH1F("h_energy", "Energy Distribution", 100, 0, 100)
    
    # Fill histogram
    for event in tree:
        hist.Fill(event.energy)
    
    # Draw
    canvas = TCanvas("c1", "Analysis", 800, 600)
    hist.Draw()
    canvas.SaveAs(f"{module}_analysis.pdf")
    
    # Cleanup
    f.Close()

if __name__ == "__main__":
    analyze_{module}()
'''
            else:
                content = "// Empty file"
            
            file_path.write_text(content)
    
    print(f"Created {sum(len(files) for files in modules.values())} files in {len(modules)} modules")
    print(f"\nTo test the extraction script, run:")
    print(f"python scripts/extract_fairship_root_usage.py --fairship-path {tmpdir}")
    print(f"\nMock FairShip will be at: {tmpdir}")
    
    return tmpdir


if __name__ == "__main__":
    create_mock_fairship()
