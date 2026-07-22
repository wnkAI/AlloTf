"""Mock TransferSamples for exercising the training loop before real data exists. They deliberately
VARY the residue count and ligand-atom count and cover all functional classes and several ligands, so
collate, masking, the sampler and the device path are all stressed. Every mock passes validate().
"""
import torch

from ..bridge.transfer_sample import TransferSample
from ..bridge.validate_sample import validate

# feature dims matching the real bridge extractors (protein 31, edge 2, atom 24, bond 3, comm 7, phys 7)
DIMS = dict(PN=31, PE=2, AN=24, BN=3, CE=7, PHYS=7, DT=4)


def _mask(n, idx):
    m = torch.zeros(n, dtype=torch.bool); m[list(idx)] = True; return m


def make_mock(n_res, n_atoms, ligand_id, y_class, seed=0, dims=DIMS, with_kd=True,
              scaffold_id="MOCK", family="MOCK_family"):
    g = torch.Generator().manual_seed(seed)
    n = n_res
    pocket = range(0, max(2, n // 8))
    dbd = range(n - 8, n)
    distal = range(n // 2, n)
    conf = torch.zeros(n); conf[list(distal)] = 0.9
    phys = torch.randn(dims["PHYS"], generator=g)
    labels = dict(functional_class=y_class, ligand_binding=1.0 if y_class != 4 else 0.0)
    if with_kd and y_class == 0:
        labels.update(dna_kd_apo=-2.0, dna_kd_ligand=0.5)
    lab_mask = {k: True for k in labels}

    ne = max(2, n * 3)
    na_e = max(2, n_atoms * 2)
    s = TransferSample(
        sample_id="%s_%d" % (scaffold_id, seed), scaffold_id=scaffold_id, candidate_id="c%d" % seed,
        ligand_id=ligand_id,
        residue_features=torch.randn(n, dims["PN"], generator=g), residue_vectors=torch.zeros(n, 0, 3),
        residue_positions=torch.randn(n, 3, generator=g) * 10,
        protein_edge_index=torch.randint(0, n, (2, ne), generator=g),
        protein_edge_features=torch.randn(ne, dims["PE"], generator=g),
        wt_residue_ids=torch.randint(0, 20, (n,), generator=g),
        mutant_residue_ids=torch.randint(0, 20, (n,), generator=g), mutation_mask=_mask(n, [1, 2]),
        pocket_mask=_mask(n, pocket), pocket_exit_mask=_mask(n, range(max(2, n // 8), n // 6 + 1)),
        hinge_mask=_mask(n, range(n // 3, n // 3 + 3)), dimer_interface_mask=_mask(n, range(n // 2, n // 2 + 3)),
        dbd_mask=_mask(n, dbd), dna_contact_mask=_mask(n, range(n - 5, n)), distal_mask=_mask(n, distal),
        ligand_atom_features=torch.randn(n_atoms, dims["AN"], generator=g),
        ligand_edge_index=torch.randint(0, n_atoms, (2, na_e), generator=g),
        ligand_edge_features=torch.randn(na_e, dims["BN"], generator=g), ligand_coordinates=torch.zeros(0, 3),
        cross_edge_index=torch.zeros(2, 0, dtype=torch.long), cross_edge_features=torch.zeros(0, 1),
        communication_edge_index=torch.randint(0, n, (2, ne), generator=g),
        communication_edge_features=torch.randn(ne, dims["CE"], generator=g),
        physics_aux=phys, physics_aux_names=tuple("p%d" % i for i in range(dims["PHYS"])),
        physics_aux_confidence=torch.rand(dims["PHYS"], generator=g),
        physics_aux_mask=torch.rand(dims["PHYS"], generator=g) > 0.3,
        native_response=torch.randn(n, dims["DT"], generator=g), native_response_mask=_mask(n, distal),
        native_response_confidence=conf,
        apo_repression=torch.tensor(float(labels.get("apo_repression", float("nan")))),
        ligand_response=torch.tensor(float(labels.get("ligand_response", float("nan")))),
        dna_kd_apo=torch.tensor(float(labels.get("dna_kd_apo", float("nan")))),
        dna_kd_ligand=torch.tensor(float(labels.get("dna_kd_ligand", float("nan")))),
        ligand_binding=torch.tensor(float(labels.get("ligand_binding", float("nan")))),
        functional_class=torch.tensor(int(y_class)),
        apo_repression_label_mask=torch.tensor(False), ligand_response_label_mask=torch.tensor(False),
        dna_kd_apo_label_mask=torch.tensor("dna_kd_apo" in lab_mask),
        dna_kd_ligand_label_mask=torch.tensor("dna_kd_ligand" in lab_mask),
        ligand_binding_label_mask=torch.tensor("ligand_binding" in lab_mask),
        functional_class_label_mask=torch.tensor(True),
        provenance={"mock": True, "family": family, "scaffold_id": scaffold_id})
    validate(s)
    return s


# mock multi-scaffold set: TtgR is deliberately huge (to check it does NOT dominate the sampler);
# the others are small/weak, and each scaffold has its own ligands and family.
_SCAFFOLDS = [
    ("TtgR", "TetR_like", ["phenol", "naringenin", "resveratrol"], 60),
    ("QacR", "TetR_like", ["rhodamine", "malachite"], 12),
    ("LacI_like", "LacI_family", ["IPTG", "GlcNAc"], 10),
    ("MarR_like", "MarR_family", ["salicylate"], 8),
]


def mock_specs():
    """Varied cross-scaffold set. Returns a list of TransferSamples spanning 4 scaffolds / 3 families,
    all functional classes, several ligands, and jittered residue / atom counts."""
    out, seed = [], 0
    for scaffold, family, ligands, count in _SCAFFOLDS:
        for k in range(count):
            out.append(make_mock(n_res=40 + (k % 6) * 6, n_atoms=12 + (k % 4) * 3,
                                 ligand_id=ligands[k % len(ligands)], y_class=k % 5, seed=seed,
                                 scaffold_id=scaffold, family=family))
            seed += 1
    return out
