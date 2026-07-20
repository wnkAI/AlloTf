"""Test 2 - native switch recovery on a scaffold with a known mechanism."""
import pytest


@pytest.mark.skip(reason="TODO(B): wire pipeline.allostery.native_template")
def test_tetr_transduction_path_recovered():
    """TetR: torsion redistribution must flag alpha6 (~103-109) and alpha4 (~49-62), and must NOT
    keep terminal residues (5, 205) after the network filter."""
