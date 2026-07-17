"""Generic aTF structure-availability layer: for any TF in the ligand DB, resolve UniProt and
query the PDB for apo / holo / DNA-bound states -> Tier. Runs for any TF, not a hand-made list.
"""
import os, csv, json, time, urllib.request, urllib.parse

DD=r"E:\DATA\AlloTf\data"
SEARCH="https://search.rcsb.org/rcsbsearch/v2/query"; GRAPHQL="https://data.rcsb.org/graphql"
JUNK={"HOH","SO4","GOL","EDO","PO4","CL","NA","K","MG","CA","ZN","MN","ACT","PEG","PG4","DMS","TRS",
 "IOD","BR","NO3","FMT","EPE","MPD","BME","CAC","1PE","P6G","2PE","PGE","CO","NI","CD","FE","CU","SCN",
 "AZI","MRD","OLC","OLA","IMD","MES","BEZ","ACY","CO3","F","LI","RB","CS","SR","BA","EOH","IPA","UNL","UNX","HEZ"}

# UniProt accessions already resolved in earlier PDB scans (authoritative; avoids gene-name ambiguity)
KNOWN={"TetR":"P0ACT4","QacR":"P0A0N4","TtgR":"Q9AIU0","RamR":"Q8ZR43","EthR":"P9WMC1",
 "LacI":"P03023","PurR":"P0ACP7","RbsR":"P0ACQ0","GalR":"P03024","GalS":"P25748","TreR":"P36673",
 "CytR":"P0ACN7","Cra":"P0ACP1","XylR_Eco":"P0ACI3","XylR_Bsu":"P94490","AraC":"P0A9E0",
 "NagC":"P0AF20","NagR":"O34817","KstR":"P96856","DesT":"Q9HUS3","FasR":"O05858","AibR":"Q1D4I5",
 "EilR":"E3G817","AvaR1":"Q82H41"}
# gene-name fallback (gene, organism keyword)
FALLBACK={"Mlc":("mlc","Escherichia coli"),"ScrR":("scrR","Streptococcus"),"MelR":("melR","Escherichia coli"),
 "RhaR":("rhaR","Escherichia coli"),"XylS":("xylS","Pseudomonas putida"),"MarR":("marR","Escherichia coli"),
 "OhrR":("ohrR","Bacillus subtilis"),"HucR":("hucR","Deinococcus radiodurans"),"GntR":("gntR","Escherichia coli"),
 "FadR":("fadR","Escherichia coli"),"BenM":("benM","Acinetobacter"),"CatM":("catM","Acinetobacter"),
 "BmrR":("bmrR","Bacillus subtilis"),"CueR":("cueR","Escherichia coli"),"ZntR":("zntR","Escherichia coli"),
 "PbrR":("pbrR","Cupriavidus"),"MerR":("merR","Shigella"),"PsiR":("psiR","Agrobacterium"),
 "HrtR":("hrtR","Lactococcus"),"QdoR":("qdoR","Bacillus subtilis"),"CmeR":("cmeR","Campylobacter")}

def getj(req,t=40):
    for _ in range(3):
        try:
            b=urllib.request.urlopen(req,timeout=t).read()
            if b.strip(): return json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code==204: return None
        except Exception: pass
        time.sleep(1.2)
    return None

def resolve_uniprot(tf):
    if tf in KNOWN: return KNOWN[tf]
    if tf not in FALLBACK: return ""
    g,org=FALLBACK[tf]
    q='gene:%s AND reviewed:true AND organism_name:"%s"'%(g,org)
    j=getj("https://rest.uniprot.org/uniprotkb/search?query="+urllib.parse.quote(q)+"&fields=accession&size=1&format=json",30)
    if j and j.get("results"): return j["results"][0]["primaryAccession"]
    j=getj("https://rest.uniprot.org/uniprotkb/search?query="+urllib.parse.quote('gene:%s AND reviewed:true'%g)+"&fields=accession&size=1&format=json",30)
    return j["results"][0]["primaryAccession"] if (j and j.get("results")) else ""

def pdb_states(acc):
    q={"query":{"type":"terminal","service":"text","parameters":{
        "attribute":"rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        "operator":"exact_match","value":acc}},"return_type":"entry",
        "request_options":{"paginate":{"rows":300},"results_verbosity":"compact"}}
    j=getj(urllib.request.Request(SEARCH,data=json.dumps(q).encode(),headers={"Content-Type":"application/json"}))
    if not j: return [],[],[],[]
    ids=[x if isinstance(x,str) else x.get("identifier") for x in j.get("result_set",[])]
    apo=[];holo=[];dna=[];tern=[]
    for i in range(0,len(ids),40):
        ql='{entries(entry_ids:%s){rcsb_id polymer_entities{entity_poly{rcsb_entity_polymer_type}} nonpolymer_entities{nonpolymer_comp{chem_comp{id}}}}}'%json.dumps(ids[i:i+40])
        jj=getj(urllib.request.Request(GRAPHQL,data=json.dumps({"query":ql}).encode(),headers={"Content-Type":"application/json"}),90)
        if not jj: continue
        for e in jj["data"]["entries"]:
            pt={p.get("entity_poly",{}).get("rcsb_entity_polymer_type") for p in (e.get("polymer_entities") or [])}
            L=[n["nonpolymer_comp"]["chem_comp"]["id"] for n in (e.get("nonpolymer_entities") or [])
               if n["nonpolymer_comp"]["chem_comp"]["id"] not in JUNK]
            hd="DNA" in pt; hl=bool(L)
            if hd and hl: tern.append(e["rcsb_id"])
            elif hd: dna.append(e["rcsb_id"])
            elif hl: holo.append(e["rcsb_id"])
            else: apo.append(e["rcsb_id"])
        time.sleep(0.2)
    return apo,holo,dna,tern

def tier(apo,holo,dna,tern):
    has_dna = bool(dna or tern); has_lig = bool(holo or tern)
    if has_dna and has_lig: return 1, "experimental apo/holo/DNA available"
    if has_lig:             return 2, "holo only - DNA state needs family homology model"
    if has_dna or apo:      return 3, "no ligand-bound structure"
    return 3, "no structure"

tfs=[]
with open(os.path.join(DD,"atf_ligand_db.csv")) as f:
    for r in csv.DictReader(f):
        if r["tf_name"] not in tfs: tfs.append(r["tf_name"])
print("resolving structures for %d transcription factors\n"%len(tfs))
rows=[]
for tf in tfs:
    acc=resolve_uniprot(tf)
    if not acc:
        rows.append([tf,"","0","0","0","0",3,"no UniProt resolved","",""]); print("%-10s  no UniProt"%tf); continue
    apo,holo,dna,tern=pdb_states(acc)
    t,note=tier(apo,holo,dna,tern)
    rows.append([tf,acc,len(apo),len(holo),len(dna),len(tern),t,note,
                 ";".join((dna+tern)[:4]),";".join(holo[:4])])
    print("%-10s %-8s apo=%-3d holo=%-3d dna=%-3d ternary=%-3d  TIER %d  %s"%(tf,acc,len(apo),len(holo),len(dna),len(tern),t,note))
    time.sleep(0.2)
out=os.path.join(DD,"atf_structure_db.csv")
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["tf_name","uniprot","n_apo","n_holo","n_dna","n_ternary","tier","tier_note","dna_pdbs","holo_pdbs"])
    w.writerows(rows)
from collections import Counter
print("\nTier distribution:",dict(Counter(r[6] for r in rows)))
print("wrote",out)
