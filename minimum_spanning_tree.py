from timeit import default_timer as timer
import argparse

import numpy as np

from graph_construction import pseudoatom_positions, atom_interaction_matrix


def mst(
    inter_mat: np.ndarray[tuple[int, int], np.dtype[int | float]],
    coords: np.ndarray[tuple[int, int], np.dtype[int | float]] | None = None,
    silent: bool = False,
    pymol: bool = False,
    start: int = 0,
    dist_th: int | float | None = None,
) -> list[tuple[int, int]]:
    """calculate the minimum spanning tree (mst) using an interaction matrix
    using Prim's algorithm
    :parameter
        - coords
          coordinates of all vertices (only needed if pymol == True)
        - inter_mat:
          (distance-) matrix specifying distances between interacting vertices
          (np.inf for not interacting ones)
          example if A-C (distance of 2) and A-B (distance of 3) would be connected
          inter_mat = np.asarray(
          [[np.inf, 3, 2], [3, np.inf, np.inf], [2, np.inf, np.inf]]
          )
        - silent:
          False to print output during search
        - pymol:
          True to print pymol mst visualization commands
        - start:
          index of the starting vertex
        - dist_th:
          to restrict connections in the inter_mat to a certain distance (all longer
          distances get set to np.inf)
    :return
        - connections
          tuples for connected vertices in the mst
    """
    if dist_th is not None:
        inter_mat[inter_mat > dist_th] = np.inf
    start_time = timer()
    # number of vertices
    num_vert = inter_mat.shape[0]
    if not silent:
        print(f"Total number of vertices: {num_vert}")
    # indices for each vertex
    vert_ind = np.arange(num_vert)
    # track already visited vertices
    visited = np.ones((num_vert,)).astype(bool)
    # set first vertex to 0
    visited[start] = False
    connections = []
    while len(inter_mat[visited]) > 0:
        # which rows should be considered
        visited_inverted = np.invert(visited)
        # get column of all visited and rows of all unvisited vertices in a matrix
        to_consider = inter_mat[visited][:, visited_inverted]
        # find the closest distance of still possible connections
        closest = np.unravel_index(np.argmin(to_consider), to_consider.shape)
        # index of vertices with the closest distance
        index_sol = (
            vert_ind[visited][closest[0]],
            vert_ind[visited_inverted][closest[1]],
        )
        # edge distance
        dist = to_consider[closest]
        # add only edges that are connected
        if dist != np.inf:
            connections.append(index_sol)
        # mark added vertex as visited
        next_vert_ind = vert_ind[visited][closest[0]]
        visited[next_vert_ind] = False
        if not silent:
            print(f"closest connection between node {index_sol[0]} and {index_sol[1]}")
            print(f"edge distance: {dist:0.4f}")
            print(f"{np.sum(visited_inverted)} of {num_vert} nodes checked")
            print(f"time elapsed: {timer() - start_time:>7.5f}")
            print("-" * 40)
    if not silent:
        print("*** all nodes checked ***")
        # check for unconnected vertices
        unconnected = vert_ind[np.invert(np.isin(vert_ind, np.unique(connections)))]
        num_unc = unconnected.shape[0]
        if num_unc > 0:
            print(f"Number of unconnected vertices: {num_unc}")
            print(f"unconnected vertices: {'-'.join(unconnected.astype(str))}")
    # print pymol commands to show connections
    if pymol:
        coords = coords.round(3).astype(str)
        for ci, i in enumerate(connections):
            t0_name = str(i[0]) + "_" + str(i[1])
            t1_name = str(i[1]) + "_" + str(i[0])
            print(
                f"pseudoatom tmpPoint{t0_name}, resi=40, chain=ZZ, b=40,"
                f"color=tv_blue, pos=[{', '.join(coords[i[0]])}]"
            )
            print(
                f"pseudoatom tmpPoint{t1_name}, resi=40, chain=ZZ, b=40,"
                f"color=tv_blue, pos=[{', '.join(coords[i[1]])}]"
            )
            print(f"distance d{ci}, tmpPoint{t0_name}, tmpPoint{t1_name}")

        print("group mst, d* tmpPoint*")

    return connections


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
        default=0,
        help="index of the start residue of the search",
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
    parser.add_argument(
        "-d",
        "--dist_th",
        type=float,
        required=False,
        default=None,
        help="max distance in \u212B between residues to be seen as interacting",
    )

    args = parser.parse_args()

    if not args.nearest_atom:
        _, coords, mat = pseudoatom_positions(args.filepath)
    else:
        _, coords, mat = atom_interaction_matrix(args.filepath)

    d = {
        "inter_mat": mat,
        "coords": coords,
        "start": args.start_residue,
        "silent": args.silent,
        "pymol": args.pymol,
        "dist_th": args.dist_th,
    }
    return d


if __name__ == "__main__":
    mst(**arg_dict())
