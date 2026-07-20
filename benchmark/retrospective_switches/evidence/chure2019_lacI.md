# Evidence: Chure et al 2019 (LacI single mutants, IPTG)

Source: Chure G, Razo-Mejia M, Belliveau NM, Einav T, Kaczmarek ZA, Barnes SL, Lewis M, Phillips R.
"Predictive shifts in free energy couple mutations to their phenotypic consequences." PNAS 2019;
116(37):18275-18284. Open access, PMC6744869. DOI: 10.1073/pnas.1907869116

Retrieved via PubMed + PMC open-access full text. Inducer = IPTG. WT LacI is inducible by IPTG.
Numbering matches the mature LacI used in our structures (Q291 = inducer-pocket residue, the same
position as the wet-lab Q291F->GlcNAc retargeting).

The value of this source for the gate: it separates two failure axes cleanly and quantitatively
(full dose-response fits), with the authors showing DNA-binding-domain mutations shift only DNA
affinity while inducer-binding-domain mutations shift only the allosteric parameters.

## Inducer-binding-domain mutations (allosteric axis)

| mutation | authors' finding                                                        | label / subtype        | grade |
|----------|-------------------------------------------------------------------------|------------------------|-------|
| Q291K    | "active state can no longer bind inducer"; inactive state preferred      | non_switch/nonresponder| A     |
| Q291R    | "abrogated inducibility outright", KA≈KI                                 | non_switch/nonresponder| A     |
| Q291V    | weakens inducer binding to BOTH active and inactive states              | non_switch/nonresponder| B     |
| F161T    | diminished inducer binding; alters KA, KI, ΔεAI                          | non_switch/nonresponder| B     |

Q291K / Q291R are the clean core-binary negatives: they fold and bind DNA (repress) but IPTG no
longer switches them. Q291V / F161T weaken rather than abolish induction -> grade B (sensitivity).

## DNA-binding-domain mutations (DNA-affinity axis; allostery intact)

| mutation | authors' finding                                       | label / subtype    | grade |
|----------|--------------------------------------------------------|--------------------|-------|
| Q18M     | strengthens DNA binding to -15.43 kBT (~O1), inducible  | functional_switch  | B     |
| Y17I     | weakens DNA affinity to -9.9 kBT (~O3, weak repression) | functional_switch  | C     |
| Q18A     | weakens DNA affinity to -11.0 kBT                        | functional_switch  | C     |

These retain IPTG allostery (still switch), so they are positives, not dna_defective negatives - none
abolishes DNA binding. Y17I/Q18A net switch phenotype is operator-dependent (leaky at weak operators)
-> grade C, sensitivity only. A clean dna_defective/constitutive NEGATIVE must come from a source that
reports loss of repression (Tack 2021 / Markiewicz I- class), not from these affinity-tuning mutants.

## Not yet covered by this source
- constitutive / dead / inverted / band-stop negatives at scale -> Tack et al 2021 (NIST, open
  access: Mol Syst Biol 10.15252/msb.202010179 and 10.15252/msb.202110847).
- a second and third scaffold (TetR, TrpR or another aTF) from public data.
