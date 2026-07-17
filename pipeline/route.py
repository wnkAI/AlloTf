"""Route: target molecule -> ranked aTF scaffolds + design mode.

Chemistry score (how similar to a native effector) and structure score (whether the scaffold is
actually designable) are computed and reported SEPARATELY - a chemically perfect hit with no
structure is not designable, and the caller must see that.

usage:  python route.py "naringenin"
        python route.py "CC(=O)Oc1ccccc1C(=O)O" --top 8
import:  from route import route ; hits = route("quinine")
"""
import os, sys, csv, urllib.request, urllib.parse
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')

DD=r"E:\DATA\AlloTf\data"
W_CHEM=dict(s2d=0.30, smcs=0.25, spharm=0.30, sphys=0.15)
W_TOTAL=dict(chem=0.6, struct=0.4)
MODE_ENH=0.95; MODE_NEAR=0.40
TIER_SCORE={1:0.70, 2:0.40, 3:0.10}

def resolve(q):
    if Chem.MolFromSmiles(q): return q
    try:
        u="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/IsomericSMILES/TXT"%urllib.parse.quote(q)
        return urllib.request.urlopen(u,timeout=25).read().decode().strip().split("\n")[0]
    except Exception: return None

def _load():
    lig=[]
    with open(os.path.join(DD,"atf_ligand_db.csv")) as f:
        for r in csv.DictReader(f):
            if r["status"]!="OK": continue
            m=Chem.MolFromSmiles(r["smiles"])
            if m is None: continue
            r["mol"]=m; lig.append(r)
    st={}
    p=os.path.join(DD,"atf_structure_db.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f): st[r["tf_name"]]=r
    return lig, st

def _feats(m):
    return np.array([Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m),
        Descriptors.NumHDonors(m),Descriptors.NumHAcceptors(m),Descriptors.NumRotatableBonds(m),
        rdMolDescriptors.CalcNumAromaticRings(m),Descriptors.FractionCSP3(m)],dtype=float)

def _struct_score(s):
    if not s: return 0.0, 3, "not in structure DB"
    t=int(s["tier"])
    rich=min(0.30, (int(s["n_holo"])+int(s["n_ternary"])+int(s["n_dna"]))/20*0.30)
    return TIER_SCORE[t]+rich, t, s["tier_note"]

def route(query, top_k=10):
    smi=resolve(query)
    if not smi: raise ValueError("cannot resolve molecule: %s"%query)
    target=Chem.MolFromSmiles(smi)
    lig,st=_load()
    tfp=AllChem.GetMorganFingerprintAsBitVect(target,2,2048,useChirality=True)
    tph=Generate.Gen2DFingerprint(target,Gobbi_Pharm2D.factory)
    tfe=_feats(target); ta=target.GetNumHeavyAtoms()
    F=np.array([_feats(e["mol"]) for e in lig]); rng=np.where(F.max(0)-F.min(0)>0,F.max(0)-F.min(0),1.0)
    hits=[]
    for i,e in enumerate(lig):
        m=e["mol"]
        s2d=DataStructs.TanimotoSimilarity(tfp,AllChem.GetMorganFingerprintAsBitVect(m,2,2048,useChirality=True))
        try: spharm=DataStructs.TanimotoSimilarity(tph,Generate.Gen2DFingerprint(m,Gobbi_Pharm2D.factory))
        except Exception: spharm=0.0
        try:   # ring-agnostic atoms (sugar ring vs open chain) but keep bond-order discrimination
            mcs=rdFMCS.FindMCS([target,m],timeout=5,ringMatchesRingOnly=False,completeRingsOnly=False)
            smcs=mcs.numAtoms/max(ta,m.GetNumHeavyAtoms()) if mcs.numAtoms>0 else 0.0
        except Exception: smcs=0.0
        sphys=float(np.clip(1-(np.abs(tfe-F[i])/rng).mean(),0,1))
        schem=(W_CHEM["s2d"]*s2d+W_CHEM["smcs"]*smcs+W_CHEM["spharm"]*spharm+W_CHEM["sphys"]*sphys)
        sstruct,tier,note=_struct_score(st.get(e["tf_name"]))
        hits.append(dict(tf=e["tf_name"],family=e["family"],native=e["native_ligand"],
            chem_note=e["chem_note"],s2d=s2d,smcs=smcs,spharm=spharm,sphys=sphys,
            s_chem=schem,s_struct=sstruct,tier=tier,tier_note=note,
            total=W_TOTAL["chem"]*schem+W_TOTAL["struct"]*sstruct,
            designable=(tier<=2)))
    hits.sort(key=lambda h:-h["total"])
    best2d=max(h["s2d"] for h in hits)
    mode=("ENHANCEMENT" if best2d>=MODE_ENH else
          "NEAR_NEIGHBOUR_RETARGETING" if best2d>=MODE_NEAR else "DISTANT_RETARGETING")
    return dict(query=query,smiles=smi,mode=mode,hits=hits[:top_k])

MODE_TXT={"ENHANCEMENT":"native ligand present -> optimise existing response (affinity / EC50 / dynamic range)",
 "NEAR_NEIGHBOUR_RETARGETING":"close native neighbour -> redesign pocket for the new molecule",
 "DISTANT_RETARGETING":"no close native ligand -> prefer promiscuous / large-pocket scaffolds, expect low first-round hit rate"}

def main():
    q=sys.argv[1] if len(sys.argv)>1 else "naringenin"
    k=int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 10
    r=route(q,k)
    t=Chem.MolFromSmiles(r["smiles"])
    print("TARGET : %s\nSMILES : %s\nprops  : MW=%.1f logP=%.2f HBD=%d HBA=%d TPSA=%.0f rings=%d"%(
        r["query"],r["smiles"],Descriptors.MolWt(t),Descriptors.MolLogP(t),Descriptors.NumHDonors(t),
        Descriptors.NumHAcceptors(t),Descriptors.TPSA(t),rdMolDescriptors.CalcNumRings(t)))
    print("MODE   : %s\n         %s\n"%(r["mode"],MODE_TXT[r["mode"]]))
    print("rank TF          family     native ligand              S_chem  S_struct tier designable  total")
    for i,h in enumerate(r["hits"],1):
        print("%3d  %-11s %-10s %-25s %.3f   %.3f    T%d   %-9s  %.3f"%(
            i,h["tf"],h["family"],h["native"][:25],h["s_chem"],h["s_struct"],h["tier"],
            "YES" if h["designable"] else "NO(no str)",h["total"]))
    d=[h for h in r["hits"] if h["designable"]]
    print("\ndesignable scaffolds in top-%d: %d"%(len(r["hits"]),len(d)))
    if d:
        b=d[0]; print("recommended: %s (%s / %s)  chem=%.2f struct=%.2f  %s"%(b["tf"],b["family"],b["native"],b["s_chem"],b["s_struct"],b["tier_note"]))
    else:
        print("WARNING: no designable scaffold in top hits - closest chemistry has no usable structure.")
if __name__=="__main__": main()
