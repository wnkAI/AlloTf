# How to run

The workflow is: **give a target molecule → the pipeline retargets a chosen scaffold to it and
scores the six states → it reports problems honestly** (a state it cannot build is marked
unavailable, never scored as zero). The process itself is the contribution; every stage is
fail-closed.

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

# retarget a scaffold to a target molecule (name, SMILES, or a .sdf path)
$PY allotf.py design --target "<molecule name or SMILES>" --scaffold <SCAFFOLD> --project results/run1
```

Output goes under `--project`: ranked designs, per-candidate six-state margins, functional category,
and a run manifest (git commit + structure/params hashes + seed).

## 3. Run the retrospective switch/non-switch gate (validation)

```bash
$PY -m benchmark.gate --scaffold <SCAFFOLD> --effector-smiles "<native inducer SMILES>" --out results/gate
```

This freezes the scoring contract (`frozen_config.json`), scores every manifest variant, and writes
`gate_results.json`. Populate `benchmark/retrospective_switches/manifest.csv` with variants whose
switch / non-switch labels come only from independent experiment (schema + discipline in
`benchmark/schema.py` and `benchmark/retrospective_switches/README.md`).

## Deposited-structure limitations the pipeline reports (not hides)

Some scaffolds' deposited states are missing what the six states need. The pipeline flags these
instead of emitting confident numbers:

- **A CA-only D state** (no side chains) is not scorable → D0 / D_DNA are marked unavailable until a
  full-atom operator structure is supplied (edit `reference.operator` for that scaffold in
  `config/scaffolds.yaml`).
- **An induced state that lacks the DBD** (disordered on induction) has no valid operator complex →
  the DNA-release margin is unavailable.

Until such a state is supplied, the gate scores whatever states are valid and marks the rest
unavailable — by design, not by accident.
