#!/bin/bash
# Fetch Rosetta's molfile_to_params.py (+ its rosetta_py helpers) into ~/rosetta_tools.
# The pip/conda PyRosetta wheels omit this script, but it is required to turn any ligand SDF
# into a .params residue type - a hard prerequisite for scoring every one of the six states.
# Source: the public RosettaCommons/rosetta repository (Rosetta is open source).
set -e
BASE=https://raw.githubusercontent.com/RosettaCommons/rosetta/main/source/scripts/python/public
D="${1:-$HOME/rosetta_tools}"
mkdir -p "$D/rosetta_py/io" "$D/rosetta_py/utility"
for f in molfile_to_params.py rosetta_py/__init__.py rosetta_py/io/__init__.py \
         rosetta_py/io/mdl_molfile.py rosetta_py/utility/__init__.py \
         rosetta_py/utility/r3.py rosetta_py/utility/rankorder.py; do
  curl -s -f -o "$D/$f" "$BASE/$f" && echo "  ok $f" || { echo "  FAIL $f"; exit 1; }
done
echo "molfile_to_params.py installed at $D/molfile_to_params.py"
echo "run with the PyRosetta env python; set ROSETTA_MOLFILE_TO_PARAMS to override the location."
