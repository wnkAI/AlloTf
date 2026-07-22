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
DATA_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/%s"
NONPOLY = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/%s/%s"
ASSEMBLY_CIF = "https://files.rcsb.org/download/%s-assembly1.cif"
_HEADERS = {"User-Agent": "AlloTF-pretrain/1.0"}


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
    """-> {resolution, nonpolymer_comp_ids, n_polymer_entities, assembly_count, title}."""
    d = _get(DATA_ENTRY % pdb_id).json()
    res = (d.get("rcsb_entry_info", {}).get("resolution_combined") or [None])
    return {
        "pdb_id": pdb_id,
        "resolution": res[0] if res else None,
        "nonpolymer_comp_ids": d.get("rcsb_entry_info", {}).get("nonpolymer_bound_components") or [],
        "n_polymer_entities": d.get("rcsb_entry_info", {}).get("polymer_entity_count", 0),
        "n_protein_entities": d.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
        "n_dna_entities": d.get("rcsb_entry_info", {}).get("polymer_entity_count_DNA", 0),
        "assembly_count": d.get("rcsb_entry_info", {}).get("assembly_count", 0),
        "title": d.get("struct", {}).get("title", "")}


def download_assembly_cif(pdb_id, out_path):
    """Biological assembly 1 mmCIF (functional oligomer), not the asymmetric unit."""
    r = _get(ASSEMBLY_CIF % pdb_id)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path
