import argparse
from timeit import default_timer as timer
import numpy as np
from graph_construction import (
    pseudoatom_positions,
    atom_interaction_matrix,
    build_graph,
)
from dijkstra import dijkstra, reconstruct_path


def search(
    filepath: str,
    dist_th: int | float = 10.0,
    nearest_atom: bool = False,
    start_residue: int | None = None,
    end_residue: int | None = None,
    silent: bool = False,
    pymol: bool = False,
):
    search_start = timer()
    # get residue data, their coordinates and the distance matrix
    if nearest_atom:
        rid, rcoords, dm, graph = build_graph(
            *atom_interaction_matrix(filepath), dist_th=dist_th
        )
    else:
        rid, rcoords, dm, graph = build_graph(
            *pseudoatom_positions(filepath), dist_th=dist_th
        )

    # Start and end of the path
    if any([start_residue is None, end_residue is None]):
        max_dists_inds = np.unravel_index(np.argmax(dm, axis=None), dm.shape)
    else:
        max_dists_inds = (start_residue, end_residue)

    # get shortest path
    v, d, pv = dijkstra(graph, max_dists_inds[0], max_dists_inds[1], silent=silent)
    # reconstruct the shortest path found
    nodes, dist = reconstruct_path(max_dists_inds[0], max_dists_inds[1], v, d, pv)
    search_end = timer()
    # info
    if not silent:
        print(
            f"{'* * ' * 9 + '*'}\n"
            "Total number of edges checked: "
            f"{int(np.sum((dm > 0.0) & (dm <= dist_th))):>5}"
        )
        print(f"Shortest path distance: {dist:>12.5f} \u212B")
        print(
            f"Total time elapsed: {search_end - search_start:>16.5f}\n"
            f"{'* * ' * 9 + '*'}"
        )
        print(f"Vertices in shortest path:\n{' - '.join(rid[nodes])}")
    if pymol:
        for i in rcoords[nodes].astype(str):
            print(
                "pseudoatom tmpPoint2, resi=40, chain=ZZ, b=40,"
                f"color=tv_blue, pos=[{', '.join(i)}]"
            )


def arg_dict() -> dict:
    """argparser for search
    :parameter
        - None:
    :return
        - d
          dictionary specifying all parameters for search
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--filepath", type=str, required=True, help="path to pdb file"
    )
    parser.add_argument(
        "-d",
        "--dist_th",
        type=float,
        required=False,
        default=10.0,
        help="max distance in \u212B between residues to be seen as interacting",
    )
    parser.add_argument(
        "-a",
        "--nearest_atom",
        action="store_true",
        help="set flag to use nearest atom between side chains to determine their "
        "interaction instead of their side chain pseudoatom",
    )
    parser.add_argument(
        "-sr",
        "--start_residue",
        type=int,
        required=False,
        default=None,
        help="index of the start residue of the path",
    )
    parser.add_argument(
        "-er",
        "--end_residue",
        type=int,
        required=False,
        default=None,
        help="index of the end residue of the path",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="set flag to not print information during runtime",
    )
    parser.add_argument(
        "-p",
        "--pymol",
        action="store_true",
        help="set flag to print pymol pseudoatom commands of the path vertices",
    )

    args = parser.parse_args()
    d = {
        "filepath": args.filepath,
        "dist_th": args.dist_th,
        "nearest_atom": args.nearest_atom,
        "start_residue": args.start_residue,
        "end_residue": args.end_residue,
        "silent": args.silent,
        "pymol": args.pymol,
    }
    return d


if __name__ == "__main__":
    search(**arg_dict())
