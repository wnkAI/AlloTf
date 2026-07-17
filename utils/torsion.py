"""Torsional allosteric fingerprint.

For any aTF: take its EXPERIMENTAL apo structures vs holo (effector-bound) structures as two
natural ensembles, and measure per-residue backbone/side-chain dihedral redistribution between
them with circular statistics.  No MD - the ensembles are crystallographic.

  R_torsion[i] = { d_circ(mu_apo, mu_holo) , JS(p_apo(theta), p_holo(theta)) }  for phi/psi/chi1..4

usage: python torsion.py TetR --max 10
"""
import os, sys, csv, json, math, time, urllib.request
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings; warnings.simplefilter("ignore", PDBConstructionWarning)

DD=r"E:\DATA\AlloTf\data"; SD=os.path.join(DD,"structures"); RD=r"E:\DATA\AlloTf\results"
os.makedirs(SD,exist_ok=True)
SEARCH="https://search.rcsb.org/rcsbsearch/v2/query"; GRAPHQL="https://data.rcsb.org/graphql"
ANGLES=["phi","psi","chi1","chi2","chi3","chi4"]
JUNK={"HOH","SO4","GOL","EDO","PO4","CL","NA","K","MG","CA","ZN","MN","ACT","PEG","PG4","DMS","TRS",
 "IOD","BR","NO3","FMT","EPE","MPD","BME","CAC","1PE","P6G","2PE","PGE","CO","NI","CD","FE","CU","SCN",
 "AZI","MRD","OLC","OLA","IMD","MES","BEZ","ACY","CO3","F","LI","RB","CS","SR","BA","EOH","IPA","UNL","UNX","HEZ"}

def getj(req,t=60):
    for _ in range(3):
        try:
            b=urllib.request.urlopen(req,timeout=t).read()
            if b.strip(): return json.loads(b)
        except Exception: pass
        time.sleep(1.0)
    return None

def states_for(acc):
    q={"query":{"type":"terminal","service":"text","parameters":{
        "attribute":"rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        "operator":"exact_match","value":acc}},"return_type":"entry",
        "request_options":{"paginate":{"rows":300},"results_verbosity":"compact"}}
    j=getj(urllib.request.Request(SEARCH,data=json.dumps(q).encode(),headers={"Content-Type":"application/json"}))
    ids=[x if isinstance(x,str) else x.get("identifier") for x in (j or {}).get("result_set",[])]
    apo=[];holo=[]
    for i in range(0,len(ids),40):
        ql='{entries(entry_ids:%s){rcsb_id polymer_entities{entity_poly{rcsb_entity_polymer_type}} nonpolymer_entities{nonpolymer_comp{chem_comp{id}}}}}'%json.dumps(ids[i:i+40])
        jj=getj(urllib.request.Request(GRAPHQL,data=json.dumps({"query":ql}).encode(),headers={"Content-Type":"application/json"}))
        if not jj: continue
        for e in jj["data"]["entries"]:
            pt={p.get("entity_poly",{}).get("rcsb_entity_polymer_type") for p in (e.get("polymer_entities") or [])}
            L=[n["nonpolymer_comp"]["chem_comp"]["id"] for n in (e.get("nonpolymer_entities") or [])
               if n["nonpolymer_comp"]["chem_comp"]["id"] not in JUNK]
            if "DNA" in pt: continue                 # DNA-bound handled separately
            (holo if L else apo).append((e["rcsb_id"], L))
    return apo, holo

def fetch(pdb_id):
    p=os.path.join(SD,pdb_id+".pdb")
    if not os.path.exists(p):
        try:
            d=urllib.request.urlopen("https://files.rcsb.org/download/%s.pdb"%pdb_id,timeout=60).read()
            open(p,"wb").write(d)
        except Exception: return None
    return p

def torsions(pdb_path):
    """-> {resnum: {angle: value_deg}} for the first protein chain"""
    st=PDBParser(QUIET=True).get_structure("x",pdb_path)
    model=next(iter(st))
    best=None
    for ch in model:
        n=sum(1 for r in ch if r.id[0]==" ")
        if n>20 and (best is None or n>best[1]): best=(ch,n)
    if not best: return {}
    ch=best[0]
    try: ch.atom_to_internal_coordinates()
    except Exception: return {}
    out={}
    for res in ch:
        ic=getattr(res,"internal_coord",None)
        if ic is None or res.id[0]!=" ": continue
        d={}
        for a in ANGLES:
            try:
                v=ic.get_angle(a)
                if v is not None: d[a]=float(v)
            except Exception: pass
        if d: out[res.id[1]]=d
    return out

def circ_mean(a):
    a=np.radians(np.asarray(a,dtype=float))
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
def circ_diff(x,y):
    d=math.radians(x-y)
    return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))
def circ_js(a,b,bins=12):
    """Jensen-Shannon divergence of two circular samples on [-180,180)"""
    e=np.linspace(-180,180,bins+1)
    p,_=np.histogram(np.asarray(a,dtype=float),bins=e); q,_=np.histogram(np.asarray(b,dtype=float),bins=e)
    p=p+0.5; q=q+0.5; p=p/p.sum(); q=q/q.sum(); m=0.5*(p+q)
    kl=lambda u,v: float((u*np.log2(u/v)).sum())
    return 0.5*kl(p,m)+0.5*kl(q,m)

def fingerprint(tf, max_each=10):
    st={}
    with open(os.path.join(DD,"atf_structure_db.csv")) as f:
        for r in csv.DictReader(f): st[r["tf_name"]]=r
    if tf not in st: raise SystemExit("TF not in structure DB: "+tf)
    acc=st[tf]["uniprot"]
    if not acc: raise SystemExit("no UniProt for "+tf)
    apo,holo=states_for(acc)
    print("%s (%s): %d apo, %d holo structures found"%(tf,acc,len(apo),len(holo)))
    apo=apo[:max_each]; holo=holo[:max_each]
    if len(apo)<2 or len(holo)<2:
        print("  need >=2 apo and >=2 holo for an ensemble -> cannot build torsion fingerprint"); return None
    def collect(ids):
        acc_={}
        for pid,ligs in ids:
            p=fetch(pid)
            if not p: continue
            t=torsions(p)
            if not t: continue
            acc_[pid]=t
            print("   %s  %-3d residues  ligands=%s"%(pid,len(t),",".join(ligs[:3]) if ligs else "-"))
        return acc_
    print("  --- apo ensemble ---");  A=collect(apo)
    print("  --- holo ensemble ---"); H=collect(holo)
    if len(A)<2 or len(H)<2: print("  too few parsable structures"); return None
    common=set.intersection(*[set(t) for t in A.values()]) & set.intersection(*[set(t) for t in H.values()])
    print("  residues present in all structures: %d"%len(common))
    rows=[]
    for rn in sorted(common):
        row={"resnum":rn,"n_apo":len(A),"n_holo":len(H)}
        sig=0.0
        for a in ANGLES:
            va=[t[rn][a] for t in A.values() if a in t[rn]]
            vh=[t[rn][a] for t in H.values() if a in t[rn]]
            if len(va)<2 or len(vh)<2: continue
            dmu=circ_diff(circ_mean(vh),circ_mean(va)); js=circ_js(va,vh)
            row["d_"+a]=round(dmu,1); row["js_"+a]=round(js,3)
            w=1.0 if a in ("phi","psi") else 0.7
            sig+=w*js
        row["torsion_signal"]=round(sig,3)
        rows.append(row)
    rows.sort(key=lambda r:-r["torsion_signal"])
    out=os.path.join(RD,"torsion_%s.csv"%tf)
    keys=["resnum","n_apo","n_holo","torsion_signal"]+[p+a for a in ANGLES for p in ("d_","js_")]
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in keys})
    print("\n  TOP torsion-responsive residues (apo -> holo redistribution):")
    print("  resnum  signal   d_phi  js_phi  d_psi  js_psi  d_chi1 js_chi1")
    for r in rows[:15]:
        print("  %5d   %.3f    %5s  %5s   %5s  %5s   %5s  %5s"%(r["resnum"],r["torsion_signal"],
            r.get("d_phi","-"),r.get("js_phi","-"),r.get("d_psi","-"),r.get("js_psi","-"),
            r.get("d_chi1","-"),r.get("js_chi1","-")))
    print("\n  saved %s  (%d residues)"%(out,len(rows)))
    return rows

if __name__=="__main__":
    tf=sys.argv[1] if len(sys.argv)>1 else "TetR"
    m=int(sys.argv[sys.argv.index("--max")+1]) if "--max" in sys.argv else 10
    fingerprint(tf,m)
