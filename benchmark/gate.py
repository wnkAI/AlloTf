"""The switch/non-switch gate as one runnable pipeline.

The innovation is the PROCESS, not any single number: a generator-agnostic, zero-shot functional
gate that scores known switches against known non-switches with the model frozen before any label is
seen, ranks by the WEAKEST necessary condition (RSM) rather than a weighted sum, adds a
parameter-free pocket->DBD resolvent channel, and attributes each failure to a margin. Every step is
fail-closed: a state that cannot be built honestly is reported unavailable, never scored as zero.

Run:  python -m benchmark.gate --scaffold LacI --out results/gate_lacI
      (needs the PyRosetta conda env + molfile_to_params; see benchmark/RUNNING.md)
"""
import argparse
import csv
import json
import os

from benchmark import freeze, schema, run_gate


NATIVE_SMILES = {
    # scaffold -> the native effector SMILES used to build the DL/IL states
    "LacI": "CC(C)S[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)O)O)O",   # IPTG
}


def _is_full_atom(pdb):
    """A protein state Rosetta can score must have side chains, not just a CA trace (1LBG is
    CA-only). Returns False for a CA-only or missing file so the caller marks the state unavailable
    instead of letting PyRosetta silently load an empty pose."""
    if not pdb or not os.path.exists(pdb):
        return False
    ca = cb = 0
    for line in open(pdb):
        if line.startswith("ATOM"):
            name = line[12:16].strip()
            if name == "CA":
                ca += 1
            elif name == "CB":
                cb += 1
    return ca > 0 and cb > ca * 0.3   # every non-Gly residue has a CB; a CA trace has none


def build_scaffold(scaffold, out_dir, operator_override=None):
    """Fetch/prepare the six-state templates + effector-loaded states + a shared backend.

    Returns everything score_variant needs. States that cannot be built honestly are set to None:
      * X_I_DNA when the induced state lacks the DBD (no valid induced operator complex);
      * any state whose structure is CA-only (unscorable) - reported, not faked.
    """
    import yaml
    from pipeline import structure, pose as pose_mod
    from pipeline.ligand_params import from_sdf
    from pipeline.physallo.rosetta_backend import PyRosettaBackend
    from rdkit import Chem
    from rdkit.Chem import AllChem

    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open("config/default.yaml"))
    ctx = dict(scaffold=scaffold, out_dir=out_dir, cfg=cfg,
               scaffold_config="config/scaffolds.yaml", structure_db="data/atf_structure_db.csv")
    st = structure.run(ctx)["states"]
    paths = dict(st["paths"])
    pocket = json.load(open(os.path.join(out_dir, "resolved_design_positions.json")))["design_positions"]

    smi = NATIVE_SMILES.get(scaffold)
    if not smi:
        raise RuntimeError("no native effector SMILES registered for %s" % scaffold)
    ligdir = os.path.join(out_dir, "lig"); os.makedirs(ligdir, exist_ok=True)
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    sdf = os.path.join(ligdir, "effector.sdf")
    w = Chem.SDWriter(sdf); w.write(m); w.close()
    params = from_sdf(sdf, ligdir, name="LIG", formal_charge=Chem.GetFormalCharge(Chem.MolFromSmiles(smi)))["params"]

    # place the native effector on both backbones (MCS transfer from the deposited holo crystal)
    crystal = paths.get("X_I_lig")
    built = {}
    for key, backbone in (("X_D_lig", "X_D"), ("X_I_lig", "X_I")):
        bpath = paths.get(backbone)
        if not _is_full_atom(bpath):
            built[key] = None; continue
        ps = pose_mod.generate_poses(smi, bpath, pocket, n_poses=6,
                                     native_pdb=crystal, native_resname=st.get("effector_resname"),
                                     native_smiles=smi)
        outp = os.path.join(ligdir, key + ".pdb")
        pose_mod.write_liganded_state(bpath, ps[0]["mol"], outp, resname="LIG",
                                      conf_id=ps[0].get("conf_id", -1)) if ps else None
        built[key] = outp if ps else None

    templates = {"X_D": paths.get("X_D"), "X_I": paths.get("X_I"),
                 "X_D_lig": built["X_D_lig"], "X_I_lig": built["X_I_lig"],
                 "X_D_DNA": paths.get("X_D_DNA"), "X_I_DNA": paths.get("X_I_DNA")}
    if st.get("induced_state_lacks_dbd"):
        templates["X_I_DNA"] = None      # no valid induced operator complex

    # drop any CA-only / missing state so build_six never scores an empty pose
    availability = {}
    for k, v in templates.items():
        ok = _is_full_atom(v)
        availability[k] = ok
        if not ok:
            templates[k] = None

    backend = PyRosettaBackend(score_function=cfg.get("design", {}).get("score_function", "ref2015"),
                               ligand_params=[params])
    from pipeline.design import _wt_chain_residues
    wt = {rn: name for rn, name in _wt_chain_residues(paths["X_D"], st.get("chain", "A"))
          if rn in set(pocket)}
    return dict(templates=templates, availability=availability, backend=backend, pocket=pocket,
                wt=wt, chain=st.get("chain", "A"))


def score_variant(scaf, mutation_map):
    """One variant -> the four scorers, or a reason it could not be scored. mutation_map: {resnum:AA3}."""
    from pipeline.state_builder import build_six, totals, linkage, dna_affinity

    cand = dict(scaf["wt"]); cand.update(mutation_map)
    need_D = scaf["availability"].get("X_D") and scaf["availability"].get("X_I")
    if not (need_D and scaf["availability"].get("X_D_lig") and scaf["availability"].get("X_I_lig")):
        return {"scored": False,
                "reason": "unscorable states: %s" % [k for k, ok in scaf["availability"].items() if not ok]}
    terms = build_six(scaf["backend"], cand, scaf["templates"], design_positions=scaf["pocket"],
                      second_shell=(), chain=scaf["chain"])
    tot = totals(terms)
    link = linkage(tot)
    if not link:
        return {"scored": False, "reason": "linkage unavailable (a core state failed to build)"}
    e_dna_xd = dna_affinity(tot, "D")
    # margins that need I_DNA (release) are left out honestly when I_DNA is unavailable
    return {"scored": True, "totals": tot, "dG_apo": link["dG_apo"], "dG_lig": link["dG_lig"],
            "ddG_coup": link["ddG_coup"], "e_dna_xd": e_dna_xd}


def main():
    ap = argparse.ArgumentParser(description="Run the switch/non-switch gate.")
    ap.add_argument("--scaffold", default="LacI")
    ap.add_argument("--manifest", default="benchmark/retrospective_switches/manifest.csv")
    ap.add_argument("--out", default="results/gate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    repo = os.getcwd()

    frozen = freeze.freeze(repo, os.path.join(args.out, "frozen_config.json"))
    print("scoring contract frozen:", frozen["contract_hash"])

    rows = [r for r in csv.DictReader(open(args.manifest)) if r["scaffold"] == args.scaffold]
    schema.validate_manifest(rows)
    print("variants for %s: %d" % (args.scaffold, len(rows)))

    scaf = build_scaffold(args.scaffold, args.out)
    print("state availability:", scaf["availability"])

    results = []
    for r in rows:
        mut = _parse_mutation(r["mutation"], scaf["wt"])
        res = score_variant(scaf, mut)
        results.append({"mutation": r["mutation"], "label": r["functional_label"],
                        "failure_subtype": r["failure_subtype"], "grade": r["evidence_grade"], **res})
        tag = "ddG_coup=%.2f" % res["ddG_coup"] if res.get("scored") else res.get("reason")
        print("  %-8s %-18s %s" % (r["mutation"], r["functional_label"], tag))

    json.dump(results, open(os.path.join(args.out, "gate_results.json"), "w"), indent=2, default=str)
    scored = [r for r in results if r.get("scored")]
    print("\nscored %d / %d variants; results -> %s/gate_results.json" %
          (len(scored), len(results), args.out))
    if not scored:
        print("NOTE: no variant scored. For %s this is the deposited-structure limitation "
              "(e.g. LacI 1LBG is CA-only); supply a full-atom operator structure to complete it."
              % args.scaffold)


AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
       "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
       "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def _parse_mutation(mut, wt):
    """'Q291K' -> {291: 'LYS'}; 'WT' -> {}."""
    if not mut or mut.upper() == "WT":
        return {}
    aa, rn = mut[-1], int(mut[1:-1])
    return {rn: AA3[aa.upper()]}


if __name__ == "__main__":
    main()
