![TEST](https://github.com/ugSUBMARINE/shortest-path/actions/workflows/test.yml/badge.svg)

This repository contains a program to find the shortest path (via Dijkstra's shortest path algorithm) between either the two furthest apart side chains (default) or between two selected residues in a protein structure.
To run the program the pdb file of the protein of interest is needed.
There are two ways to create the graph based on the protein structure: Either the distance between the closest side chain atom or the distance between pseudoatoms (midpoints of the catalytically important atoms in a side chain) are used to calculate connected vertices.
If needed, commands that create pseudoatoms for pymol that represent points in the shortest path can be displayed.

**Software Requirements:**
*  [Python3.10](https://www.python.org/downloads/)

*optional:*
*  [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

In order to see all parameters run:

`python3 shortest_protein_path.py -h`

To run the program in default mode run:

`python3 shortest_protein_path.py -f /PATH/TO/PDB/FILE`
