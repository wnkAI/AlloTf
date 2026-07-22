"""Kept only for the encoder's self-supervised multi-state PRETRAINING. The runtime data contract is
allotf_model.bridge.TransferSample - do not confuse the two."""
from .multistate_graph import StateGraph, MultiStateGraph

__all__ = ["StateGraph", "MultiStateGraph"]
