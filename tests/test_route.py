"""Test 1 - native ligand recovery: a native effector must return its own TF."""
import sys
sys.path.insert(0, r"E:\DATA\AlloTf")
from pipeline.route import route


def test_native_ligand_recovers_scaffold():
    r = route("naringenin", top_k=5)
    tfs = [h["tf"] for h in r["hits"]]
    assert "TtgR" in tfs, tfs
    assert r["mode"] == "ENHANCEMENT", r["mode"]


def test_chem_and_structure_reported_separately():
    r = route("quinine", top_k=5)
    h = r["hits"][0]
    assert "s_chem" in h and "s_struct" in h and "tier" in h
