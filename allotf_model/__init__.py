"""AlloTF: a multi-state, physics-constrained, end-to-end learnable allosteric-switch model.

Learns (apo, ligand, DNA-bound) -> switching directly, with a shared SE(3)-equivariant encoder over
the three states, an explicit state-difference module, three task heads (Binding / Path / Switch),
mechanistic constraints, and a multiplicative gated final score. No external pretrained model; the
encoder is trained from scratch.
"""
