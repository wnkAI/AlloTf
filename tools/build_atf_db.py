"""aTF-ligand database v1: cross-family, native effector ligands with SMILES from PubChem."""
import os, csv, time, urllib.request, urllib.parse

DD=r"E:\DATA\AlloTf\data"
# (tf_name, family, native_ligand_name, pubchem_query, note)
DB=[
 # ---- TetR family ----
 ("TetR","TetR","tetracycline","tetracycline","antibiotic"),
 ("QacR","TetR","berberine","berberine","cationic multidrug"),
 ("QacR","TetR","ethidium","ethidium","cationic multidrug"),
 ("TtgR","TetR","naringenin","naringenin","flavonoid"),
 ("TtgR","TetR","quercetin","quercetin","flavonoid"),
 ("TtgR","TetR","chloramphenicol","chloramphenicol","antibiotic"),
 ("RamR","TetR","berberine","berberine","multidrug"),
 ("EthR","TetR","hexadecyl octanoate","hexadecyl octanoate","lipid ester"),
 ("KstR","TetR","3-oxo-cholest-4-en-26-oyl-CoA","cholesterol","steroid-CoA (proxy: cholesterol)"),
 ("DesT","TetR","oleoyl-CoA","oleoyl-CoA","acyl-CoA"),
 ("FasR","TetR","palmitoyl-CoA","palmitoyl-CoA","acyl-CoA"),
 ("AibR","TetR","isovaleryl-CoA","isovaleryl-CoA","acyl-CoA"),
 ("AvaR1","TetR","gamma-butyrolactone","gamma-butyrolactone","quorum signal"),
 ("CmeR","TetR","taurocholate","taurocholic acid","bile salt"),
 ("EilR","TetR","crystal violet","crystal violet","cationic dye"),
 # ---- LacI/GalR family ----
 ("LacI","LacI","allolactose","allolactose","sugar"),
 ("LacI","LacI","IPTG","IPTG","sugar analog"),
 ("PurR","LacI","hypoxanthine","hypoxanthine","purine"),
 ("PurR","LacI","guanine","guanine","purine"),
 ("RbsR","LacI","D-ribose","D-ribose","sugar"),
 ("GalR","LacI","D-galactose","D-galactose","sugar"),
 ("GalS","LacI","D-galactose","D-galactose","sugar"),
 ("TreR","LacI","trehalose 6-phosphate","trehalose 6-phosphate","sugar-P"),
 ("CytR","LacI","cytidine","cytidine","nucleoside"),
 ("Cra","LacI","fructose 1,6-bisphosphate","fructose 1,6-bisphosphate","sugar-P"),
 ("ScrR","LacI","sucrose","sucrose","sugar"),
 # ---- ROK family ----
 ("XylR_Bsu","ROK","D-xylose","D-xylose","sugar"),
 ("NagC","ROK","N-acetylglucosamine 6-phosphate","N-acetylglucosamine 6-phosphate","amino sugar-P"),
 ("NagR","ROK","N-acetylglucosamine 6-phosphate","N-acetylglucosamine 6-phosphate","amino sugar-P"),
 ("Mlc","ROK","glucose","D-glucose","sugar"),
 # ---- AraC/XylS family ----
 ("AraC","AraC","L-arabinose","L-arabinose","sugar"),
 ("XylR_Eco","AraC","D-xylose","D-xylose","sugar"),
 ("MelR","AraC","melibiose","melibiose","sugar"),
 ("RhaR","AraC","L-rhamnose","L-rhamnose","sugar"),
 ("XylS","AraC","benzoate","benzoic acid","aromatic acid"),
 # ---- MarR family ----
 ("MarR","MarR","salicylate","salicylic acid","aromatic acid"),
 ("OhrR","MarR","cumene hydroperoxide","cumene hydroperoxide","peroxide"),
 ("HucR","MarR","urate","uric acid","purine"),
 # ---- GntR family ----
 ("GntR","GntR","D-gluconate","D-gluconic acid","sugar acid"),
 ("FadR","GntR","palmitoyl-CoA","palmitoyl-CoA","acyl-CoA"),
 # ---- LysR family ----
 ("BenM","LysR","benzoate","benzoic acid","aromatic acid"),
 ("BenM","LysR","cis,cis-muconate","muconic acid","aromatic acid"),
 ("CatM","LysR","cis,cis-muconate","muconic acid","aromatic acid"),
 # ---- MerR family ----
 ("BmrR","MerR","tetraphenylphosphonium","tetraphenylphosphonium","cationic multidrug"),
 ("CueR","MerR","Cu(I)","copper","metal"),
 ("ZntR","MerR","Zn(II)","zinc","metal"),
 ("PbrR","MerR","Pb(II)","lead","metal"),
 ("MerR","MerR","Hg(II)","mercury","metal"),
 # ---- others ----
 ("PsiR","DeoR-like","D-allulose","D-psicose","rare sugar"),
 ("HrtR","TetR","heme","heme","cofactor"),
 ("QdoR","TetR","quercetin","quercetin","flavonoid"),
]

def smiles_of(name):
    try:
        u="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/IsomericSMILES/TXT"%urllib.parse.quote(name)
        return urllib.request.urlopen(u,timeout=25).read().decode().strip().split("\n")[0]
    except Exception as e:
        return ""

out=os.path.join(DD,"atf_ligand_db.csv")
seen={}
rows=[]
for tf,fam,lig,q,note in DB:
    if q in seen: smi=seen[q]
    else:
        smi=smiles_of(q); seen[q]=smi; time.sleep(0.25)
    ok="OK" if smi else "NO_SMILES"
    rows.append([tf,fam,lig,smi,note,ok])
    print("%-10s %-9s %-32s %-4s %s"%(tf,fam,lig,ok,smi[:60]))
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["tf_name","family","native_ligand","smiles","chem_note","status"])
    w.writerows(rows)
n_ok=sum(1 for r in rows if r[5]=="OK")
print("\nwrote %s : %d entries, %d with SMILES, %d families"%(out,len(rows),n_ok,len(set(r[1] for r in rows))))
