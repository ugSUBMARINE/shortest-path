![TEST](https://github.com/ugSUBMARINE/shortest-path/actions/workflows/test.yml/badge.svg)

This repository contains two programs:
*   **shortest_protein_path** 
    * Find the shortest path (via Dijkstra's shortest path algorithm) between either the two furthest apart vertices (side chains) (default) or between two selected vertices (residues in a protein structure).
        ![alt text](https://github.com/ugSUBMARINE/shortest-path/blob/master/test_data/sp.png?raw=true)
*   **minimum_spanning_tree**
    * Find the minimum spanning tree (via Prim's minimum spanning tree algorithm) between all connected vertices (side chains with a distance lower than the threshold) using distance matrix.
        ![alt text](https://github.com/ugSUBMARINE/shortest-path/blob/master/test_data/mst.png?raw=true)

To run each program the pdb file of the protein of interest is needed.

There are two ways to create the graph based on the protein structure to calculate connected vertices: 
*   The distance between the closest side chain atoms
*   The distance between pseudoatoms (midpoints of the catalytically important atoms in a side chain) 

If needed, commands that create pseudoatoms and distances for pymol that represent points and their connections in the shortest path can be displayed.

*Each program can also be used with non- protein inputs.*

**Software Requirements:**
*  [Python3.10](https://www.python.org/downloads/)

*optional:*
*  [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

In order to see all parameters run:

`python3 shortest_protein_path.py -h`

`python3 minimum_spanning_tree.py -h`

To run the programs in default mode run:

`python3 shortest_protein_path.py -f /PATH/TO/PDB/FILE`

`python3 minimum_spanning_tree.py -f /PATH/TO/PDB/FILE`
