"""RCSB clients for the structure re-query. Search API turns a UniProt accession into every PDB id
for that protein (so apo entries the CSV never stored are recovered); Data API returns per-entry
metadata (resolution, non-polymer ligands, polymer entities, assemblies). Biological-assembly mmCIF
is downloaded, not the asymmetric unit.

Transient failures (429/503/timeout) are retried with backoff and then RAISED with a reason - never
swallowed into "this PDB has no data", which would silently shrink the dataset.
"""
import time

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
ASSEMBLY_CIF = "https://files.rcsb.org/download/%s-assembly1.cif"
_HEADERS = {"User-Agent": "AlloTF-pretrain/1.0"}

# one call per entry: resolution + entity counts + ALL ligand comp ids (reliable, unlike the
# nonpolymer_bound_components field which RCSB leaves null for many entries, e.g. 1LBH/IPTG)
_ENTRY_QUERY = """query($id:String!){ entry(entry_id:$id){
  rcsb_entry_info{ resolution_combined polymer_entity_count_protein polymer_entity_count_DNA assembly_count }
  struct{ title }
  nonpolymer_entities{ nonpolymer_comp{ chem_comp{ id } } } } }"""


def _get(url, tries=4, backoff=1.5, **kw):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = "HTTP %d" % r.status_code; time.sleep(backoff ** (k + 1)); continue
            r.raise_for_status()
        except requests.RequestException as e:
            last = str(e); time.sleep(backoff ** (k + 1))
    raise RuntimeError("RCSB GET failed after %d tries (%s): %s" % (tries, last, url))


def search_by_uniprot(uniprot, max_hits=500):
    """Every PDB id whose polymer maps to this UniProt accession (apo AND holo AND complexes)."""
    query = {
        "query": {"type": "terminal", "service": "text",
                  "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers."
                                              "reference_sequence_identifiers.database_accession",
                                 "operator": "exact_match", "value": uniprot}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_hits}}}
    last = None
    for k in range(4):
        try:
            r = requests.post(SEARCH_URL, json=query, headers=_HEADERS, timeout=30)
            if r.status_code == 204:
                return []                                    # no hits (a real answer, not an error)
            if r.status_code == 200:
                return [h["identifier"] for h in r.json().get("result_set", [])]
            if r.status_code in (429, 500, 502, 503, 504):
                last = "HTTP %d" % r.status_code; time.sleep(1.5 ** (k + 1)); continue
            r.raise_for_status()
        except requests.RequestException as e:
            last = str(e); time.sleep(1.5 ** (k + 1))
    raise RuntimeError("RCSB search failed for %s (%s)" % (uniprot, last))


def fetch_entry(pdb_id):
    """-> {resolution, nonpolymer_comp_ids, n_protein_entities, n_dna_entities, assembly_count, title}.
    Ligand comp ids come from the non-polymer ENTITIES (reliable), not nonpolymer_bound_components."""
    last = None
    for k in range(4):
        try:
            r = requests.post(GRAPHQL_URL, json={"query": _ENTRY_QUERY, "variables": {"id": pdb_id}},
                              headers=_HEADERS, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                last = "HTTP %d" % r.status_code; time.sleep(1.5 ** (k + 1)); continue
            r.raise_for_status()
        except requests.RequestException as e:
            last = str(e); time.sleep(1.5 ** (k + 1))
    else:
        raise RuntimeError("RCSB graphql failed for %s (%s)" % (pdb_id, last))
    e = (r.json().get("data") or {}).get("entry")
    if e is None:
        raise RuntimeError("RCSB returned no entry for %s" % pdb_id)
    ei = e.get("rcsb_entry_info", {})
    res = ei.get("resolution_combined") or [None]
    ligs = [((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
            for ne in (e.get("nonpolymer_entities") or [])]
    return {
        "pdb_id": pdb_id, "resolution": res[0] if res else None,
        "nonpolymer_comp_ids": [c for c in ligs if c],
        "n_protein_entities": ei.get("polymer_entity_count_protein", 0),
        "n_dna_entities": ei.get("polymer_entity_count_DNA", 0),
        "assembly_count": ei.get("assembly_count", 0),
        "title": (e.get("struct") or {}).get("title", "")}


_COMP_QUERY = """query($id:String!){ chem_comp(comp_id:$id){
  rcsb_chem_comp_descriptor{ SMILES_stereo SMILES } } }"""


def fetch_comp_smiles(comp_id):
    """Canonical SMILES for a ligand comp id (for chemotype clustering). None if unavailable."""
    for k in range(3):
        try:
            r = requests.post(GRAPHQL_URL, json={"query": _COMP_QUERY, "variables": {"id": comp_id}},
                              headers=_HEADERS, timeout=30)
            if r.status_code == 200:
                d = ((r.json().get("data") or {}).get("chem_comp") or {}).get("rcsb_chem_comp_descriptor") or {}
                return d.get("SMILES_stereo") or d.get("SMILES")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 ** (k + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(1.5 ** (k + 1))
    return None


def download_assembly_cif(pdb_id, out_path):
    """Biological assembly 1 mmCIF (functional oligomer), not the asymmetric unit."""
    r = _get(ASSEMBLY_CIF % pdb_id)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path
