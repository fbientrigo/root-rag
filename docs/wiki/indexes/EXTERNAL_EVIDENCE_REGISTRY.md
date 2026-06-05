# External Evidence Registry

This registry tracks high-quality external sources (CERN CDS, arXiv, Official Docs) used to validate or provide context for internal claims.

| Ref ID | Title | URL | Source Type | Key Data Point |
| :--- | :--- | :--- | :--- | :--- |
| **EXT-001** | SHiP Technical Proposal | [CERN-SPSC-2015-016](https://cds.cern.ch/record/2007512) | Official Proposal | UBT 0.1 GeV/c momentum cut baseline. |
| **EXT-002** | The SHiP experiment at the proposed CERN SPS Beam Dump Facility | [arXiv:2112.01487](https://arxiv.org/abs/2112.01487) | Peer-reviewed (EPJ C 82:486) | 45 MeV threshold for 30cm liquid scintillator (Section 5.3.2). |
| **EXT-003** | Muon DIS Background in SHiP | [Indico (EPFL 383457)](https://indico.cern.ch/event/383457/) | Conference Talk | Definition of Front/Side/Cavern muon regions. |
| **EXT-004** | ROOT TVectorD Documentation | [root.cern/doc/master/classTVectorD](https://root.cern/doc/master/classTVectorD.html) | Official Docs | TArrayD-based vector for linear algebra. |
| **EXT-005** | PYTHIA 8 settings for DIS | [pythia.org](https://pythia.org) | Official Docs | `WeakBosonExchange`, `dipoleRecoil`, `Q2Min`. |
| **EXT-006** | Reciprocal Rank Fusion (RRF) | [OpenSearch RRF Docs](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) | Technical Ref | Fusing multiple rankers without score normalization. |

## Usage
Add `[[EXTERNAL_EVIDENCE_REGISTRY#EXT-00X]]` to the `source` field in claim registries or wiki notes.
