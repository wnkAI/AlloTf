"""DNA-release check: does effector binding actually pull the DBD off the operator?

Pure geometry, experimental structures only - no ddG, no MD.
  1. take the TF-operator complex (DNA-bound reference)
  2. superimpose the apo ensemble and the holo (effector-bound) ensemble onto it, aligning ONLY the
     ligand-binding core (the DBD is deliberately excluded from the fit)
  3. measure where each ensemble's DBD lands relative to the DNA: contacts and clashes
  apo should sit on the operator; holo should drift off it.

usage: python dna_check.py TetR
"""
import os, sys, csv, json, time, urllib.request
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings; warnings.simplefilter("ignore", PDBConstructionWarning)

DD=r"E:\DATA\AlloTf\data"; SD=os.path.join(DD,"structures"); RD=r"E:\DATA\AlloTf\results"
os.makedirs(SD,exist_ok=True)
CONTACT=4.5; CLASH=2.2

def fetch(pid):
    p=os.path.join(SD,pid+".pdb")
    if not os.path.exists(p):
        try: open(p,"wb").write(urllib.request.urlopen("https://files.rcsb.org/download/%s.pdb"%pid,timeout=60).read())
        except Exception: return None
    return p

def load(pid):
    p=fetch(pid)
    return PDBParser(QUIET=True).get_structure(pid,p)[0] if p else None

def chains(model):
    prot=[];dna=[]
    for ch in model:
        res=[r for r in ch if r.id[0]==" "]
        if not res: continue
        names={r.get_resname().strip() for r in res}
        if names <= {"DA","DT","DG","DC","DU","A","T","G","C","U"}: dna.append(ch)
        elif len(res)>30: prot.append(ch)
    return prot,dna

def ca(ch, lo=None, hi=None):
    d={}
    for r in ch:
        if r.id[0]!=" " or "CA" not in r: continue
        n=r.id[1]
        if lo is not None and not (lo<=n<=hi): continue
        d[n]=r["CA"]
    return d

def dbd_dna_geometry(model_prot_chain, dna_atoms, dbd_lo, dbd_hi):
    """contacts/clashes between DBD residues of one protein chain and the DNA"""
    pa=[a for r in model_prot_chain if r.id[0]==" " and dbd_lo<=r.id[1]<=dbd_hi for a in r if a.element!="H"]
    if not pa or not dna_atoms: return 0,0,np.nan
    P=np.array([a.coord for a in pa]); D=np.array([a.coord for a in dna_atoms])
    d=np.linalg.norm(P[:,None,:]-D[None,:,:],axis=2)
    return int((d<CONTACT).sum()), int((d<CLASH).sum()), float(d.min())

def run(tf, dna_pdb, core_range, dbd_range, apo_ids, holo_ids):
    ref=load(dna_pdb)
    if ref is None: raise SystemExit("cannot load DNA complex "+dna_pdb)
    rp,rd=chains(ref)
    if not rd: raise SystemExit("%s has no DNA chain"%dna_pdb)
    dna_atoms=[a for ch in rd for r in ch if r.id[0]==" " for a in r if a.element!="H"]
    print("reference DNA complex %s: %d protein chains, %d DNA chains (%d DNA atoms)"%(
        dna_pdb,len(rp),len(rd),len(dna_atoms)))
    ref_chain=rp[0]; ref_core=ca(ref_chain,*core_range)
    # native DNA-bound conformation as internal control
    c,cl,mn=dbd_dna_geometry(ref_chain,dna_atoms,*dbd_range)
    print("  [control] DNA-bound reference itself: DBD-DNA contacts=%d clashes=%d min_dist=%.2f\n"%(c,cl,mn))

    out=[]
    for tag,ids in (("apo",apo_ids),("holo",holo_ids)):
        for pid in ids:
            m=load(pid)
            if m is None: continue
            pc,_=chains(m)
            if not pc: continue
            ch=pc[0]; mv=ca(ch,*core_range)
            common=sorted(set(mv)&set(ref_core))
            if len(common)<40: continue
            sup=Superimposer()
            sup.set_atoms([ref_core[i] for i in common],[mv[i] for i in common])
            sup.apply([a for r in ch for a in r])          # move whole chain by core-domain fit
            c,cl,mn=dbd_dna_geometry(ch,dna_atoms,*dbd_range)
            out.append((tag,pid,len(common),round(sup.rms,2),c,cl,mn))
            print("  %-4s %s  core_fit_rms=%.2f (%d CA)   DBD-DNA: contacts=%-4d clashes=%-3d min_dist=%.2f"%(
                tag,pid,sup.rms,len(common),c,cl,mn))
    A=[r for r in out if r[0]=="apo"]; H=[r for r in out if r[0]=="holo"]
    if A and H:
        ca_,ch_=np.mean([r[4] for r in A]),np.mean([r[4] for r in H])
        cla,clh=np.mean([r[5] for r in A]),np.mean([r[5] for r in H])
        print("\n  ================ RESULT ================")
        print("  apo  ensemble (n=%d): DBD-DNA contacts %.1f   clashes %.1f"%(len(A),ca_,cla))
        print("  holo ensemble (n=%d): DBD-DNA contacts %.1f   clashes %.1f"%(len(H),ch_,clh))
        drop=(ca_-ch_)/ca_*100 if ca_>0 else 0
        print("  contact change on effector binding: %+.1f%%"%(-drop))
        verdict=("HOLO RELEASES DNA (contacts drop / clashes rise) - coupling direction correct"
                 if (ch_<ca_ or clh>cla) else "no release detected by geometry alone")
        print("  verdict: %s"%verdict)
    with open(os.path.join(RD,"dna_check_%s.csv"%tf),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["state","pdb","n_core_CA","core_fit_rms","dbd_dna_contacts","dbd_dna_clashes","min_dist"])
        w.writerows(out)
    print("  saved results/dna_check_%s.csv"%tf)

# per-scaffold definitions (core = ligand-binding domain used for the fit; DBD excluded from fit)
CFG={
 "TetR": dict(dna_pdb="1QPI", core_range=(75,205), dbd_range=(1,50),
              apo_ids=["1A6I","1BJZ","2NS7","2XGC"], holo_ids=["2TRT","2VKE","1ORK","2X6O"]),
 "QacR": dict(dna_pdb="1JT0", core_range=(60,180), dbd_range=(1,45),
              apo_ids=["1JTY"], holo_ids=["1JUS","1JUP","1RKW"]),
}
if __name__=="__main__":
    tf=sys.argv[1] if len(sys.argv)>1 else "TetR"
    if tf not in CFG: raise SystemExit("no config for %s (need DNA-complex PDB + core/DBD ranges)"%tf)
    run(tf,**CFG[tf])
