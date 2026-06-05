# Cross-Section and Weighting

## The DIS Cross-Section ($\sigma_{DIS}$)
The total Deep Inelastic Scattering cross-section is calculated by Pythia6 during the generation phase.

- **Extraction**: Pythia variable `PARI(1)`.
- **Units**: mbarns (millibarns).
- **Storage**: 
  - Initially saved in `InMuon[10]` in the `muonDis.root` file.
  - Propagated to `MCTrack[1].GetWeight()` during transport.
  - Copied to a dedicated `CrossSection` branch in the final output by `run_simScript.py`.

## The Multiplicity Factor ($nDIS$)
To study rare DIS backgrounds, many DIS interactions are generated for a single muon. This "oversampling" must be corrected.

- **Factor**: `args.nDIS` (default is often 100 or 1000).
- **Correction**: Every event weight is multiplied by $1/nDIS$.

## Material Budget Weight ($w_{material}$)
The probability of a DIS interaction depends on the density and thickness of the material the muon passes through.

- **Calculation**: $\text{MeanMaterialBudget}$ along the muon's path.
- **Weight**: $\rho_{average} \cdot L$ (Density $\times$ Length).

## Final Physics Normalization
To get the number of expected events for a given POT, use:
$$N_{phys} = \sum \frac{W_{muon} \cdot \sigma_{DIS} \cdot w_{material}}{nDIS \cdot nmuons}$$

Where:
- $W_{muon}$: Original muon weight.
- $nmuons$: Number of muons in the original MuonBack event.

## Evidence Anchors
- `muonDIS/makeMuonDIS.py:227`: `myPythia.GetPARI(1)`.
- `docs/wiki/fairship/normalization/MuDIS_normalization_factors.md`: Formal normalization summary.
- `macro/run_simScript.py:841-860`: Post-processing to add `CrossSection` branch.
