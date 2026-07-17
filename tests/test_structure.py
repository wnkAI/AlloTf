"""QC must REJECT, not silently pass, when a state is unusable."""
import pytest


@pytest.mark.skip(reason="TODO(B): implement pipeline.structure.prepare")
def test_qc_rejects_additive_as_effector():
    ...


@pytest.mark.skip(reason="TODO(B)")
def test_qc_rejects_wrong_oligomer():
    ...
