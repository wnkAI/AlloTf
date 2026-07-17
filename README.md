# AlloTF-RL

**Mechanism-constrained closed-loop redesign of allosteric transcription factors from sparse cellular dose–time fluorescence.**

Give it a target molecule. It picks a native aTF scaffold, generates candidates by physics, and a contextual-bandit reinforcement learner refines the next batch from a handful of in-cell dose–time fluorescence measurements — about a dozen plasmids per target.

## Pipeline

```
target molecule
   → pick a native aTF scaffold                       (Route)
   → prepare D / I conformers, operator, ligand pose   (Structure)
   → extract THAT scaffold's allosteric template       (Allostery — no training, no MD)
   → PhysAllo: aa_filter → rotamer enumeration → sequence + rotamer search
   → six-state PyRosetta ref2015:  D0  I0  DL  IL  D_DNA  I_DNA
   → compute: target binding · ddG coupling · apo DNA competence
              DNA release · allosteric similarity · decoy specificity
   → hard gates → Pareto → diversity                   (Rank)
   → 8 plasmids → in-cell dose–time fluorescence
   → contextual-bandit update → 4 new plasmids
```

**12 plasmids per target (8 + 4). Not 96, not thousands.**

## What it optimises

Not "does the ligand bind hardest". The frozen objective is that the target ligand **selectively
stabilises the induced state**, the apo state **keeps holding the operator**, and the induced state
**lets it go**.

| quantity | six-state form | want |
|---|---|---|
| ddG coupling | `(E_IL − E_DL) − (E_I0 − E_D0)` | < 0 |
| apo DNA competence | `E(D·DNA) − E(D0)` | binding retained |
| DNA release | `[E(I·DNA) − E(I0)] − [E(D·DNA) − E(D0)]` | weakened (double difference) |
| decoy specificity | `E(best decoy) − E(target)` | target wins |

Every number is a relative proxy on one scaffold — not a Kd, not a general allostery model, not an
MD result. The point is not per-timepoint accuracy; it is to reach a functional sensor with ~12
plasmids instead of hundreds.

> Design does not optimise whether the target ligand binds most strongly; it optimises whether
> target binding **selectively stabilises the transcriptionally active allosteric state**.

## The learner

A **mechanism-conditioned hierarchical Gaussian-process contextual bandit** — reinforcement
learning and active learning in one, not deep RL and not hundreds of training points. It learns the
raw fluorescence surface

```
(sequence, PhysAllo mechanism features, log c, t)  →  F(c, t)
```

directly, without first fitting rate/lag constants (unstable at a 6 h read cadence). B, fold
induction, EC50, the first-response time bin, and response AUC are read off the predicted surface.
Selection is multi-objective posterior Thompson sampling — no fixed weighted score — with basal
leak as a hard constraint. Scaffolds share the mechanism weights but each carries a
scaffold-specific random effect, `f_global(mechanism) + f_scaffold`, so TtgR is one task among
several rather than a stand-in for every TF.

## Commands

```bash
# 1. first batch
python allotf.py design --target target.sdf --initial-designs 8
#    → initial_8.fasta  initial_8_plate_layout.csv  initial_8_features.csv

# 2. import the plate
python allotf.py feedback --project target_project --plate fluorescence.csv
#    columns: candidate_id, concentration, time_h, fluorescence, replicate

# 3. next batch
python allotf.py select-next --project target_project --n 4
#    → next_4.fasta  next_4_plate_layout.csv  posterior_predictions.csv
```

Plate design per target: 8 candidates × 5 concentrations (0, 0.1, 0.3, 1, 3 × Cref) × 2 replicates
= 80 wells, plus WT and empty vector — one 96-well plate. Reads at 0, 6, 12, 18, 24, 30, 36 h
(repeated reads of the same wells, so no extra plasmids). A second bandit-chosen batch of 4
follows.

## Layout

```
allotf.py                 CLI: design | feedback | select-next
config/    default · scoring · forcefield
data/      atf_ligand_db.csv · atf_structure_db.csv · tables/ (Sensor-seq QC, engine check only)
pipeline/  route · structure · allostery · design · state_builder · rank
   physallo/  aa_filter · rotamers · sidechain · search · scoring (fast clash prefilter) · rosetta_backend (atomic energy)
   ai/        fluorescence · response_gp · bandit · acquisition · experiment_io · closed_loop
utils/     standardize_ligand · residue_mapping · torsion · contacts · energy
tools/     build_atf_db · build_structure_db · build_tables
tests/     gates · multistate · structure · allostery · route
```

## Validation

1. **Native-ligand recovery** — feed a native effector, Route returns its own TF; the WT sequence scores high.
2. **Native switch direction** — on WT, the native effector must *weaken* the predicted DNA interface. A scaffold that cannot reproduce the WT direction must not enter design.
3. **Known-mutant separation** — the hard gates must demote known constitutive / DNA-binding-defective mutants.

## Hard-won facts — read before you code

These were each verified the expensive way. Do not re-learn them.

- **PubChem `CanonicalSMILES` silently drops stereochemistry.** D-ribose, D-xylose and L-arabinose all collapsed to Tanimoto **1.000**. Use `IsomericSMILES`, and RDKit fingerprints need `useChirality=True` (they ignore stereo by default). → `utils/standardize_ligand.py`
- **Sugars: ring vs open chain wrecks 2D similarity.** Cyclic native xylose vs open-chain xylitol scored **0.03** on ECFP, while open-chain gluconate scored 0.43 — the wrong scaffold won. Never route sugars on 2D fingerprints alone.
- **Enrichment metrics: break ties randomly.** A constant predictor scored **EF@50 = 9.0** because tied scores were cut in file order and the source table is sorted by activity (first-50 sensor density was **101×** background). With tie-shuffling it returns to 1.02. Unit-test every metric with random / constant / permuted-label / perfect predictors.
- **Molecular descriptors do not generalise across chemotypes.** Chemotype-holdout Spearman ≈ **0.18**; pooled seq+descriptor (0.447) was *worse* than per-ligand sequence models (0.495) — negative transfer. This is why the platform conditions on **structure**, not on a learned chemistry mapping.
- **MD does not resolve this allostery.** A Nature Biotech 2026 study found 2.1 μs × 3 replicas *inconclusive* for their own switch; only HDX-MS was interpretable. The platform therefore uses **crystallographic ensembles**, not simulation.
- **Torsion fingerprints work.** On TetR (8 apo vs 8 holo crystal structures), per-residue φ/ψ/χ circular redistribution recovered the textbook path unaided: **five of the top-15 residues (103,104,106,108,109) fall in α6** (8.6× enrichment over chance) and **49, 62 fall in α4** — the pendulum-motion helix and the helix that pushes the DBDs apart.
- **…but graph distance alone cannot localise transduction.** A null model (random pocket/DBD labels) puts ~137/207 residues "on-path", so no residue is graph-significant. Transduction is called from three independent structural signals — torsion change AND contact-network churn AND reachability — not from graph membership.
- **DBD readout uses the operator-contacting recognition residues, not the DBD centroid.** On TetR the centroid moves 0.53 Å (reads as "nothing happened") while the recognition helix swings 3.55 Å — the centroid averages a pivot away.
- **Always load the biological assembly.** 1QPI stores its functional dimer as two MODEL blocks that reuse chain 'A'; reading the first model silently collapses the dimer to a monomer and the contact graph loses the entire dimer interface.
- **Sensor-seq is an engine check, not the subject.** It confirms sequence→response is learnable (per-ligand ρ ≈ 0.495) and gives an optional functional prior for TtgR only. It cannot supply cross-scaffold allostery.
- **"Has a structure" ≠ "has a usable allosteric state."** Tier must be QC'd: is the bound ligand a real effector or a crystallisation additive? is the DNA a real operator? is the oligomer right? `structure.py:qc()` **rejects**, it does not warn.

## Structure tiers (auto-resolved, `data/atf_structure_db.csv`)

| tier | meaning | count |
|---|---|---|
| 1 | experimental apo/holo/DNA | 15 (TetR, QacR, LacI, PurR, BmrR, NagR, MarR, BenM, KstR, AibR, AvaR1 …) |
| 2 | apo+holo, DNA state needs family homology model | 6 (TtgR, RamR, EthR, AraC, XylR_Eco, CatM) |
| 3 | not designable | 24 |

17 scaffolds have ≥2 apo **and** ≥2 holo — the requirement for a torsion template.
