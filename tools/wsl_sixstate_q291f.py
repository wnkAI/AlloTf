"""Prove the pipeline runs: real PyRosetta ref2015 six-state scoring of LacI WT vs Q291F for GlcNAc.

WT should NOT switch on GlcNAc (Q291 does not fit it); Q291F should give a more favourable
multistate linkage proxy. This is the smallest end-to-end demonstration that the six states build
and score on a real backend - not a claim of free energy, just that the machine turns.
"""
import os, sys
sys.path.insert(0, "/mnt/e/DATA/AlloTf")
os.chdir("/mnt/e/DATA/AlloTf")
import numpy as np
from pipeline.structure import fetch_assembly, load_assembly, classify_chains, write_state, resolve_design_positions
from pipeline import pose as pose_mod
from pipeline.ligand_params import _parameterise_mol
from pipeline.physallo.rosetta_backend import PyRosettaBackend
from pipeline.state_builder import build_six, totals, linkage

OUT = "/mnt/e/DATA/AlloTf/results/_q291f_wsl"; os.makedirs(OUT, exist_ok=True)
GLCNAC="CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@@H]1O)CO)O)O"; IPTG="CC(C)S[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)O)O)O"

print("=== build backbones (1LBH holo) ===", flush=True)
holo=fetch_assembly("1LBH",OUT); ch=[c.id for c in classify_chains(load_assembly(holo))[0]][:2]
X_I=write_state(holo,os.path.join(OUT,"X_I.pdb"),ch)
X_I_ipt=write_state(holo,os.path.join(OUT,"X_I_lig_native.pdb"),ch,drop_het=True,keep_ligand="IPT")
pocket,_=resolve_design_positions(holo,"IPT",dbd_range=[1,61],cutoff=5.0)
print("pocket:",pocket,flush=True)

print("=== place GlcNAc + parameterise ===", flush=True)
ps=pose_mod.generate_poses(GLCNAC,X_I,pocket,n_poses=6,native_pdb=X_I_ipt,native_resname="IPT",native_smiles=IPTG)
X_I_lig=pose_mod.write_liganded_state(X_I,ps[0]["mol"],os.path.join(OUT,"X_I_lig.pdb"),resname="TGT",conf_id=ps[0].get("conf_id",-1))
tgt=_parameterise_mol(ps[0]["mol"],OUT,"TGT","target",conf_id=ps[0].get("conf_id",-1),smiles=GLCNAC)
print("params:",tgt["params"],"charge",tgt["formal_charge"],flush=True)

# minimal 4-state (no DNA states - LacI IPTG structure lacks DBD, documented): D0 I0 DL IL
be=PyRosettaBackend(score_function="ref2015",ligand_params=[tgt["params"]])
templates={"X_D":X_I,"X_I":X_I,"X_D_lig":X_I_lig,"X_I_lig":X_I_lig}  # single-backbone proof-of-run
wt_res={p:load_assembly(X_I) and None for p in []}  # placeholder
# read WT residues at pocket
m=load_assembly(X_I); chA=max(classify_chains(m)[0],key=lambda c:sum(1 for r in c if r.id[0]==" "))
wt={r.id[1]:r.get_resname().upper() for r in chA if r.id[0]==" "}
def cand(mut): 
    d={p:wt[p] for p in pocket}; d.update(mut); return d

print("=== six-state (4 usable) build: WT ===", flush=True)
for tag,mut in [("WT",{}),("Q291F",{291:"PHE"})]:
    terms=build_six(be, cand(mut), templates, pocket, chain="A")
    tot=totals(terms); link=linkage(tot)
    print("%-6s totals=%s"%(tag,{k:round(v,1) if v else v for k,v in tot.items()}),flush=True)
    print("       linkage=%s"%({k:round(v,2) for k,v in link.items()} if link else None),flush=True)
print("DONE",flush=True)
