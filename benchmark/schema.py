"""Manifest schema + validation for the retrospective switch/non-switch gate.

Loud on purpose: a mislabelled or under-documented variant is worse than a missing one, because it
silently corrupts a zero-shot benchmark whose whole value is that the labels never touched the model.
Every row must carry its experimental provenance and grade; unknown enums are rejected, not coerced.
"""

REQUIRED = [
    "scaffold", "family", "operator", "ligand",
    "wt_sequence", "variant_sequence", "mutation",
    "functional_label", "failure_subtype",
    "assay_type", "basal_output", "induced_output", "fold_change", "ligand_concentration",
    "evidence_source", "evidence_grade",
]

FUNCTIONAL_LABEL = {"functional_switch", "non_switch"}

# negatives carry one of these; positives use "" (no failure)
FAILURE_SUBTYPE = {
    "",                              # functional_switch
    "constitutive",
    "nonresponder",
    "dna_defective",
    "binding_without_switching",
    "nonbinder",
    "decoy_responsive",
    "folding_expression_defective",  # kept OUT of the core binary (see README)
}

EVIDENCE_GRADE = {"A", "B", "C"}

# which margin the weakest-link should point at for each failure subtype (attribution check).
# Keys must be members of FAILURE_SUBTYPE; m_release is exercised via binding_without_switching
# (binds ligand and DNA but never releases the operator), which should fall on link/release.
EXPECTED_WEAKEST = {
    "constitutive": "apo",
    "nonresponder": "lig",          # also acceptable: link
    "dna_defective": "dna",
    "decoy_responsive": "spec",
    "binding_without_switching": "link",
}

# subtypes that must not enter the core switch/non-switch binary
EXCLUDE_FROM_CORE_BINARY = {"folding_expression_defective"}


def _num_or_blank(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def validate_row(row, i=None):
    """row: dict. Raises ValueError on the first problem; returns True if clean."""
    where = "" if i is None else " (row %d, %s)" % (i, row.get("mutation", "?"))
    for c in REQUIRED:
        if c not in row:
            raise ValueError("missing column '%s'%s" % (c, where))

    if row["functional_label"] not in FUNCTIONAL_LABEL:
        raise ValueError("functional_label '%s' not in %s%s"
                         % (row["functional_label"], sorted(FUNCTIONAL_LABEL), where))
    if row["failure_subtype"] not in FAILURE_SUBTYPE:
        raise ValueError("failure_subtype '%s' not in %s%s"
                         % (row["failure_subtype"], sorted(FAILURE_SUBTYPE), where))
    if row["evidence_grade"] not in EVIDENCE_GRADE:
        raise ValueError("evidence_grade '%s' not A/B/C%s" % (row["evidence_grade"], where))

    lbl, sub = row["functional_label"], row["failure_subtype"]
    if lbl == "functional_switch" and sub != "":
        raise ValueError("a functional_switch must have empty failure_subtype%s" % where)
    if lbl == "non_switch" and sub == "":
        raise ValueError("a non_switch must name a failure_subtype%s" % where)

    if not row.get("evidence_source"):
        raise ValueError("evidence_source is mandatory - a label with no citation is not admissible%s"
                         % where)
    for c in ("basal_output", "induced_output", "fold_change", "ligand_concentration"):
        if not _num_or_blank(row.get(c)):
            raise ValueError("%s must be numeric or blank, got %r%s" % (c, row.get(c), where))
    return True


def in_core_binary(row):
    """Grade A/B and not an expression/folding failure -> counts in the main switch/non-switch test."""
    return (row["evidence_grade"] in ("A", "B")
            and row["failure_subtype"] not in EXCLUDE_FROM_CORE_BINARY)


def validate_manifest(rows):
    """Validate every row and return summary counts. Raises on the first invalid row."""
    for i, r in enumerate(rows):
        validate_row(r, i)
    scaffolds = {}
    for r in rows:
        scaffolds.setdefault(r["scaffold"], {"switch": 0, "non_switch": 0})
        key = "switch" if r["functional_label"] == "functional_switch" else "non_switch"
        scaffolds[r["scaffold"]][key] += 1
    core = sum(1 for r in rows if in_core_binary(r))
    return {"n": len(rows), "n_core_binary": core, "scaffolds": scaffolds}
