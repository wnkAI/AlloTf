"""Produce PAIRED design candidates so a gain-tuning mutation can be judged against its own recognition
baseline:

    P1        recognition mutations only (does the new ligand use the native pathway at all?)
    P1-G1..k  the SAME recognition mutant + ONE constrained transduction mutation at a high-bottleneck
              gain-tuning-shell residue (retunes amplification without touching the pocket)

Keeping the pair lets the wet-lab attribute a change in ligand-induced DNA release to the gain mutation
itself, not to expression / stability / baseline DNA binding. Generators (OpenDDE/Protenix/Rosetta)
build structures for these; they never decide function.
"""
# conservative transduction substitutions - small polarity/rigidity changes, not pocket contacts
_TUNING_AA = ("ALA", "SER", "GLY", "LEU", "VAL")


def make_gain_variants(recognition_mutations, gain_tuning_residues, bottleneck_score, k=2,
                       aa_choices=_TUNING_AA):
    """recognition_mutations: {ci: AA}. gain_tuning_residues: allowed canonical indices (the gain shell).
    bottleneck_score: {ci: float} or tensor indexable by ci. -> [(label, mutations_dict), ...] with the
    recognition-only baseline first."""
    ranked = sorted(gain_tuning_residues, key=lambda i: -float(bottleneck_score[i]))
    base = dict(recognition_mutations)
    variants = [("P1", dict(base))]
    for j, ci in enumerate(ranked[:k], start=1):
        mut = dict(base)
        mut[ci] = aa_choices[j % len(aa_choices)]       # one constrained transduction mutation
        variants.append(("P1-G%d" % j, mut))
    return variants
