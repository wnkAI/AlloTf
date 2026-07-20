# Retrospective switch / non-switch gate

A zero-shot external functional validation set. It asks one question, with no model trained on these
labels:

> With RSM's WT calibration, sign directions and gates **frozen before any mutant is seen**, do the
> six-state margins and a **parameter-free** pocket->DBD resolvent gain separate known functional
> switches from known non-switches — and do they attribute the right failure mode?

Four scorers are compared on the same variants: `binding_only`, `RSM`, `g_resolvent`, `RSM+g`.

## Non-negotiable discipline

1. **Labels are independent of computation.** A variant's `functional_label` / `failure_subtype`
   comes only from experiment: dose-response, reporter assay, published phenotype, binding data, or
   the authors' explicit classification. It is **never** inferred from a Rosetta score, from whether
   a residue looks pocket-lining, or from our own allosteric-path call. Doing so would be circular.

2. **Freeze, then unblind.** Order is fixed: freeze the RSM formula + thresholds + sign conventions
   (`frozen_config` with a hash) → freeze this dataset → run the real six states → analyse **once**.
   No look-at-labels → retune-weights → re-report on the same variants.

3. **If tuning is truly needed**, split into a `development` set and a `locked_test` set, and split
   **by scaffold** — never scatter similar mutants of one scaffold across both. `frozen_split.json`
   records the split and the config hash it was frozen at.

4. **Keep the failure classes.** Not a plain active/inactive binary. Negatives carry a
   `failure_subtype`, because the gate's real test is whether the weakest margin points at the right
   mechanism, not just whether a binary classifier works.

5. **Evidence grade gates the analysis.** A = full dose-response or quantitative phenotype; B =
   single-concentration functional assay; C = qualitative only. Main results use A and B; C is
   sensitivity analysis only. `folding_expression_defective` never enters the core binary — it may
   reflect expression, not an allosteric-mechanism failure.

## What each failure subtype should light up (the attribution check)

| experimental phenotype        | expected weakest margin |
|-------------------------------|-------------------------|
| constitutive                  | `m_apo`                 |
| nonresponder                  | `m_lig` / `m_link`      |
| dna_defective                 | `m_dna`                 |
| no_dna_release                | `m_release`             |
| decoy_responsive              | `m_spec`                |
| binding_without_switching     | `m_link` (bind ok, no switch) |

A scorer that classifies switch/non-switch adequately but attributes failure to the wrong margin has
limited methodological value — that is reported, not hidden.

## Frozen decision rules (set before results)

- **RSM works, resolvent adds gain** → NBT physics line continues; RIFT protein arm is greenlit.
- **RSM works, resolvent adds nothing** → NBT continues; RIFT stays on synthetic/mechanical domains;
  `g` is not forced into the AlloTF ranking.
- **Resolvent works, RSM weak** → re-examine six-state signs / DNA-state build / WT calibration /
  ligand pose before any training; do not train RIFT to paper over a physics-definition problem.
- **Neither separates** → stop expanding the RIFT protein arm and any large prospective design;
  audit labels, state fidelity, and whether negatives are really expression/folding failures.

## Files

- `manifest.csv` — one row per variant (schema in `benchmark/schema.py`)
- `evidence/` — per-source notes / extracted phenotypes with citations
- `structures/` — fetched PDBs and built six-state models
- `frozen_split.json` — dev/locked-test split + config hash
- `rsm_scores.parquet`, `resolvent_scores.parquet` — computed scores
- `report.html` — the one-shot unblinded analysis
