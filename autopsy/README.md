# LigandMPNN Autopsy

**Nothing here assumes LigandMPNN is irreplaceable.** It is on probation until these six probes
report. The design pipeline treats it as one swappable backend (`pipeline/design.py`), never as
the designer.

## Why

LigandMPNN optimises

```
P(S | X_backbone, X_ligand)      loss = -sum_i log P(a_i_native | X, L)
```

i.e. *"which residue does the PDB usually put in this geometry?"* — not *"which sequence binds this
target and switches the output"*. `sequence recovery ⇏ binding`, and certainly `⇏ allosteric
switching`.

The single most damning clue is **in the paper's own ablation**: removing ligand **element type**
barely changes near-ligand sequence recovery. If erasing chemical identity does not move the
metric, the metric is not measuring chemistry — and a model trained/selected on it may be reading
the pocket shape while the ligand is little more than a placeholder volume.

Add: **no negative data at all** (the PDB contains successful complexes only — it never learned why
a sequence should *not* bind, nor how to reject a decoy or the wrong stereoisomer), fixed pose (a
wrong pose gets an excellent sequence *for the wrong pose*), and side-chain accuracy that decays
outward (χ1 84% → χ4 19%) exactly where salt bridges and directional H-bonds are made.

## The six probes

| # | probe | question | fail signal |
|---|---|---|---|
| 1 | **ligand deletion** | true vs deleted vs random vs coord-shuffled ligand | `KL < 0.10` → designs from the backbone, not the ligand |
| 2 | **atom-identity counterfactual** | freeze all coords, change ONLY chemistry (C→N/O, neutral→charged, donor→acceptor, single→aromatic) | `<5%` of pocket positions react → blind to chemistry, sees shape |
| 3 | **pose perturbation** | shift/rotate ligand 0.25 / 0.5 / 1.0 / 2.0 Å | flat confidence → **confidently wrong on wrong poses** |
| 4 | **stereochemistry** | correct isomer vs enantiomer vs epimer vs isostere | `KL < 0.05` → stereo-blind (fatal for sugars/polyols) |
| 5 | **vs ProteinMPNN** | same pocket, model that never sees the ligand | `agreement > 80%` → the ligand channel is near-redundant |
| 6 | **function vs recovery** | does its log-likelihood separate real Sensor-seq labels? | `AUC ≤ 0.60` → uninformative about function |

Thresholds are fixed in `probes.py:THR` **before** running, so the conclusion cannot be
rationalised after seeing the numbers.

### Probe 6 is NOT structure-free (an earlier draft of this file was wrong)

LigandMPNN consumes backbone **and** ligand coordinates. Probe 6 therefore needs **one canonical
TtgR backbone** plus a **uniform, comparably built pose for each of the 9 ligands**. No new
experimental structures are required, but if the 9 poses are built differently the probe measures
*pose quality differences between ligands*, not model ability.

It must also:
* score **only the 11 design positions** (67, 70, 74, 78, 89, 92, 93, 96, 110, 113, 114) — a
  whole-sequence LL is swamped by the ~200 residues identical across all 16,191 variants;
* compare **WT-relative**: `dLL(S,L) = LL_pocket(S,L) − LL_pocket(WT,L)`;
* and report the metric that actually matters — the **ligand-conditioning increment** over a model
  that never sees the ligand:

```
I(S,L) = LL_LigandMPNN(S|X,L) − LL_ProteinMPNN(S|X)
```

If `I(S,L)` carries no functional signal, the ligand channel adds nothing.

**Reference already in hand:** our own per-ligand sequence model reaches ρ ≈ 0.495 on this data
using nothing but 11-position one-hot. A PDB-scale "ligand-aware" network that cannot beat that
has no claim on the ranking.

## The three possible verdicts → what we do

| verdict | meaning | action |
|---|---|---|
| **A** | ligand chemistry genuinely drives it | keep encoder+decoder as the sequence prior; bolt on multistate physics + allosteric template |
| **B** | it runs on geometry / backbone | keep the protein encoder, **rewrite the ligand encoder and the protein–ligand interaction module** |
| **C** | score unrelated to function | demote to a plausibility filter; **keep it out of the final ranking entirely** |

**Probe 6 alone can force verdict C.** It needs no new *experimental* structures — we already hold
16,191 QC'd TtgR variants with measured transcriptional response — but it does need the canonical
backbone + 9 uniform poses described above.

## What it is benchmarked against

`PhysAlloDesignBackend` is the production designer; `PhysicsDesignBackend` is its single-state
ablation. LigandMPNN is neither — it is the **baseline**, and `production=False` is enforced in
code (`design.py:run` refuses to generate candidates with it). The physics backends score with
explicit terms

```
E = E_LJ + E_Coulomb + E_HB(directional) + E_solvation + E_strain + E_unsat_polar
```

over a **chemically complete** ligand (formal/partial charge, bond order, aromaticity,
hybridisation, protonation + tautomer microstates, chirality) — precisely the information the
element ablation suggests LigandMPNN may be ignoring. *(Wording: an empirical force field is
**physics-grounded**, not "first-principles" in the quantum sense. Do not overclaim.)*

If the physics backend matches or beats the neural prior on probe 6, the prior's role shrinks to
foldability.

## Run

```bash
pip install torch          # already present
git clone https://github.com/dauparas/LigandMPNN && bash get_model_params.sh
python autopsy/run_autopsy.py --scaffold TetR --out autopsy/report_TetR.json
```

All six probes need the frozen checkpoint. Probe 6 additionally needs one canonical TtgR backbone
+ 9 uniformly built ligand poses (see above) - it is not structure-free.

Diagnostic probes for dissecting LigandMPNN pocket log-likelihoods.
