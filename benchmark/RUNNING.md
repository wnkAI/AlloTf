# How to run

The workflow is: **give a target molecule → the pipeline retargets LacI to it and scores the six
states → it reports problems honestly** (a state it cannot build is marked unavailable, never scored
as zero). The process itself is the contribution; every stage is fail-closed.

## 1. One-time environment (WSL, conda)

```bash
# PyRosetta lives in its own conda env (the pip wheel omits the ligand param tool)
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n rosetta -y python=3.11
conda activate rosetta
conda install -y -c https://conda.rosettacommons.org pyrosetta
conda install -y -c conda-forge rdkit biopython scipy scikit-learn pandas pyarrow
pip install pyyaml requests

# the ligand parameter tool (pip/conda PyRosetta both omit it) - fetch once from the public Rosetta source
bash tools/get_molfile_to_params.sh          # installs into ~/rosetta_tools
```

Always run with this env's python: `~/miniconda3/envs/rosetta/bin/python`.

## 2. Run the process on a molecule

```bash
cd /path/to/AlloTf
PY=~/miniconda3/envs/rosetta/bin/python

# retarget LacI to a molecule (name, SMILES, or a .sdf path)
$PY allotf.py design --target "N-acetyl-D-glucosamine" --scaffold LacI --project results/glcnac
# or by SMILES:
$PY allotf.py design --target "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@@H]1O)CO)O)O" --scaffold LacI
```

Output goes under `--project`: ranked designs, per-candidate six-state margins, functional category,
and a run manifest (git commit + structure/params hashes + seed).

## 3. Run the retrospective switch/non-switch gate (validation)

```bash
$PY -m benchmark.gate --scaffold LacI --out results/gate_lacI
```

This freezes the scoring contract (`frozen_config.json`), scores every manifest variant, and writes
`gate_results.json`. Add variants in `benchmark/retrospective_switches/manifest.csv` (schema +
discipline in `benchmark/schema.py` and `benchmark/retrospective_switches/README.md`).

## Known LacI structural limitation (the process reports it, does not hide it)

LacI's deposited states are each missing something the six states need, so the pipeline flags them
instead of emitting confident numbers:

- **D state (1LBG) is CA-only** (no side chains) → D0 / D_DNA are not scorable as deposited.
- **Induced state (1LBH) lacks the DBD** (disordered on induction) → no valid I_DNA → the DNA-release
  margin is unavailable.

To get complete LacI numbers, point the D state at a **full-atom** LacI-operator structure (edit the
`reference.operator` entry for LacI in `config/scaffolds.yaml`). Until then the gate scores whatever
states are valid and marks the rest unavailable — by design, not by accident.
```
