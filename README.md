# AlloTF V1

**Target molecule in → ranked customised allosteric transcription factor designs out.**

```
target molecule
   → pick a native aTF scaffold          (Route)
   → extract THAT system's native allosteric template   (no training, no MD)
   → generate target-conditioned pocket sequences        (LigandMPNN proposals)
   → static multi-state negative design                  (four states per candidate)
   → rank                                                (hard gates → Pareto → diversity)
```

## What V1 is and is not

**Is:** a machine for *enriching* designs that satisfy four conditions simultaneously —

| condition | score | want |
|---|---|---|
| target ligand prefers the induced state | `S_state = E_L(X_D) − E_L(X_I)` | > 0 |
| apo state still binds the operator | `E_DNA(X_D)` | low |
| induced state no longer binds the operator | `S_release = E_DNA(X_I) − E_DNA(X_D)` | > 0 |
| the native distal allosteric response survives | template similarity | high |

**Is not:** a general allostery model, an MD pipeline, or a Kd predictor. **Every number is a
relative proxy on one scaffold.** The platform is not expected to be right every time; it is
expected to compress the search space from ~10⁴ blind variants to ~10² testable designs.

> Design does not optimise whether the target ligand binds most strongly; it optimises whether
> target binding **selectively stabilises the transcriptionally active allosteric state**.

## Layout

```
allotf.py                 main entry
config/  default | scoring | forcefield
data/    atf_ligand_db.csv  (51 ligands × 9 families)
         atf_structure_db.csv (45 TFs, auto-resolved Tier)
         tables/            Sensor-seq QC tables (engine check only)
pipeline/ route · structure · allostery · pose · design · state_builder
          ligand_score · dna_release · specificity · rank · active_learning
utils/    standardize_ligand · residue_mapping · torsion · contacts · energy
tools/    build_atf_db · build_structure_db · build_tables · gen_skeleton
tests/    test_route · test_structure · test_allostery · test_dna_release · test_end_to_end
```

## Team split

| owner | modules | acceptance |
|---|---|---|
| **A** | `route.py`, `standardize_ligand.py`, databases, decoys | native ligand and its analogue both return sensible scaffolds + mode |
| **B** | `structure.py`, `allostery.py`, `residue_mapping.py` | on a classic TF, path traced pocket → DBD, reproduces the known change direction |
| **C** | `pose.py`, `design.py` | mask-constrained, structurally sane, sequence-diverse candidates |
| **D** | `state_builder.py`, `ligand_score.py`, `dna_release.py`, `specificity.py`, `energy.py`, `contacts.py` | native effector drives the predicted DNA interface in the *weakening* direction on a known inducible TF |
| **E** | `rank.py`, `active_learning.py`, `allotf.py`, reporting | SMILES → final designs with no manual file passing |

## Three system tests before V1 ships

1. **Native ligand recovery** — feed a native effector, Route returns its own TF; WT sequence scores high.
2. **Native switch direction** — on WT, the native effector must *weaken* the predicted DNA interface. **A scaffold that cannot reproduce the WT direction must not enter design.**
3. **Known mutant separation** — the hard gates must demote known constitutive / DNA-binding-defective mutants.

## Hard-won facts — read before you code

These were each verified the expensive way. Do not re-learn them.

- **PubChem `CanonicalSMILES` silently drops stereochemistry.** D-ribose, D-xylose and L-arabinose all collapsed to Tanimoto **1.000**. Use `IsomericSMILES`, and RDKit fingerprints need `useChirality=True` (they ignore stereo by default). → `utils/standardize_ligand.py`
- **Sugars: ring vs open chain wrecks 2D similarity.** Cyclic native xylose vs open-chain xylitol scored **0.03** on ECFP, while open-chain gluconate scored 0.43 — the wrong scaffold won. Never route sugars on 2D fingerprints alone.
- **Enrichment metrics: break ties randomly.** A constant predictor scored **EF@50 = 9.0** because tied scores were cut in file order and the source table is sorted by activity (first-50 sensor density was **101×** background). With tie-shuffling it returns to 1.02. Unit-test every metric with random / constant / permuted-label / perfect predictors.
- **Molecular descriptors do not generalise across chemotypes.** Chemotype-holdout Spearman ≈ **0.18**; pooled seq+descriptor (0.447) was *worse* than per-ligand sequence models (0.495) — negative transfer. This is why the platform conditions on **structure**, not on a learned chemistry mapping.
- **MD does not resolve this allostery.** A Nature Biotech 2026 study found 2.1 μs × 3 replicas *inconclusive* for their own switch; only HDX-MS was interpretable. V1 therefore uses **crystallographic ensembles**, not simulation.
- **Torsion fingerprints work.** On TetR (8 apo vs 8 holo crystal structures), per-residue φ/ψ/χ circular redistribution recovered the textbook path unaided: **five of the top-15 residues (103,104,106,108,109) fall in α6** (8.6× enrichment over chance) and **49, 62 fall in α4** — the pendulum-motion helix and the helix that pushes the DBDs apart.
- **…but terminal residues are noise.** Residues 5 and 205 also ranked high — they just wobble. Torsion signal is only meaningful **after** filtering to residues on the pocket→DBD network.
- **Sensor-seq is an engine check, not the subject.** It confirms sequence→response is learnable (per-ligand ρ ≈ 0.495) and provides an optional functional prior for TtgR only. It cannot supply cross-scaffold allostery.
- **"Has a structure" ≠ "has a usable allosteric state."** Tier must be QC'd: is the bound ligand a real effector or a crystallisation additive? is the DNA a real operator? is the oligomer right? `structure.py:qc()` must **reject**, not warn.

## Structure tiers (auto-resolved, `data/atf_structure_db.csv`)

| tier | meaning | count |
|---|---|---|
| 1 | experimental apo/holo/DNA | 15 (TetR, QacR, LacI, PurR, BmrR, NagR, MarR, BenM, KstR, AibR, AvaR1 …) |
| 2 | apo+holo, DNA state needs family homology model | 6 (TtgR, RamR, EthR, AraC, XylR_Eco, CatM) |
| 3 | not designable | 24 |

17 scaffolds have ≥2 apo **and** ≥2 holo — the requirement for a torsion template.

## Run

```bash
python allotf.py --target naringenin --top-scaffolds 3
python allotf.py --target xylitol --objective retarget --raw-designs 10000 --final-designs 96
```

Stages report their owner and stop cleanly at the first unimplemented module.
