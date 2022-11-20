import numpy as np
from matplotlib import pyplot as plt

X_DICT = {
    "ALA": ["CB"],
    "CYS": ["SG"],
    "ASP": ["OD2", "OD1"],
    "GLU": ["OE1", "OE2"],
    "PHE": ["CG", "CD1", "CE1", "CD2", "CE2", "CZ"],
    "GLY": ["CA"],
    "HIS": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "ILE": ["CB", "CG1", "CD1", "CG2"],
    "LYS": ["NZ"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "MET": ["SD", "CE"],
    "ASN": ["OD1", "ND2"],
    "PRO": ["N", "CA", "CB", "CG", "CD"],
    "GLN": ["OE1", "NE2"],
    "ARG": ["NE", "CZ", "NH1", "NH2"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "VAL": ["CB", "CG1", "CG2"],
    "TRP": ["CE3", "CZ3", "CH2", "CZ2", "CE2", "NE1", "CD1", "CG", "CD2"],
    "TYR": ["OH"],
}


def dist_calc(
    arr1: np.ndarray[tuple[int, int], np.dtype[int | float]],
    arr2: np.ndarray[tuple[int, int], np.dtype[int | float]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    calculates euclidean distances between all points in two k-dimensional arrays
    'arr1' and 'arr2'
        :parameter
            - arr1: N x k array
            - arr2: M x k array
        :return
            - dist: M x N array with pairwise distances
    """
    norm_1 = np.sum(arr1 * arr1, axis=1).reshape(1, -1)
    norm_2 = np.sum(arr2 * arr2, axis=1).reshape(-1, 1)

    dist = (norm_1 + norm_2) - 2.0 * np.dot(arr2, arr1.T)
    # necessary due to limited numerical accuracy
    dist[dist < 1.0e-11] = 0.0

    return np.sqrt(dist)


def pseudoatom_positions(
    target_pdb_file: str | None = None,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[str]],
    np.ndarray[tuple[int, int], np.dtype[float]],
]:
    """calculates pseudoatom positions for all residues in target_pdb_file
        which is the mean distance between all atoms according to X_DICT
     :parameter
        - X_DICT: every amino acid with its catalytically important atoms eg
          {"ASP": ["OD2", "OD1"],...}
        - target_pdb_file:
          pdb file with data of protein of interest
    :return
        pseudo_data: 2D list like [[Res 3letter, ChainID, ResidueID],...]
        pseudo_coords: 2D list like [[pseudo_x1, pseudo_y1, pseudo_z1],...]"""
    # list of all data of the entries like
    # [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # read all lines
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data += [
                [
                    line[12:16].replace(" ", ""),
                    line[17:20].replace(" ", ""),
                    line[21].replace(" ", ""),
                    line[22:26].replace(" ", ""),
                ]
            ]
            res_coords += [[line[30:38], line[38:46], line[46:54]]]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)

    aa = list(X_DICT.keys())
    # list of 2D arrays where each 2D array is like
    # [[Res 3letter, ChainId, ResidueID],...]
    # each 2D array is for an aa
    pseudo_data = []
    # list of 2D arrays where each 2D array is like
    # [[pseudo_x1, pseudo_y1, pseudo_z1],...]
    # each 2D array is for an aa
    pseudo_coords = []
    for i in aa:
        # where aa i is located i res_data and res_coords
        ind_in_ori_arr = np.where(res_data[:, 1] == i)[0]
        # if aa i exists
        if len(ind_in_ori_arr) > 0:
            # which atom is a catalytically important atom
            cat_imp_at_ind = np.isin(res_data[ind_in_ori_arr][:, 0], X_DICT[i])

            # coordinates[where aa in res_data/coords][catalytically important atom]
            cat_imp_at_coords = res_coords[ind_in_ori_arr][cat_imp_at_ind]
            cat_im_at_data = res_data[ind_in_ori_arr][cat_imp_at_ind]

            # to get for each residue one entry with [Res 3letter, ChainID, ResID]
            pseudo_data += np.asarray(
                np.split(cat_im_at_data, len(cat_imp_at_coords) / len(X_DICT[i]))
            )[:, :, 1:4][:, 0].tolist()

            # pseudo coordinates for all residues of aa i
            pseudo_coords += np.mean(
                np.asarray(
                    np.split(
                        cat_imp_at_coords,
                        len(cat_imp_at_coords) / len(X_DICT[i]),
                    )
                ),
                axis=1,
            ).tolist()

    pseudo_data = np.asarray(pseudo_data)
    pseudo_coords = np.asarray(pseudo_coords)

    # remove duplicated side chain entries and store only their first appearing
    rd_un, rd_uc = np.unique(pseudo_data, axis=0, return_index=True)
    rd_uc = np.sort(rd_uc)
    pseudo_data = pseudo_data[rd_uc]
    pseudo_coords = pseudo_coords[rd_uc]

    # sort the residues again by chain and residue index
    ori_sort = np.lexsort((pseudo_data[:, 2].astype(int), pseudo_data[:, 1]))
    pseudo_data = pseudo_data[ori_sort]
    pseudo_coords = pseudo_coords[ori_sort]

    return pseudo_data, pseudo_coords, dist_calc(pseudo_coords, pseudo_coords)


def data_coord_extraction(
    target_pdb_file: str,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[any]],
    np.ndarray[tuple[int, 3], np.dtype[float]],
]:
    """reads the pdb file and stores all coordinates and the residue data - changes
    *** CA of GLY to CB ***
    :parameter
         - target_pdb_file:
           path to pdb file for protein of interest
    :returns
         - new_data:
           contains information about all residues like
           [[Atom type, Residue 3letter, ChainID, ResidueID],...]
         - new_coords:
           contains coordinates of corresponding residues to the new_data entries
    """
    # list of all data of the entries like
    # [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data += [
                [
                    line[12:16].replace(" ", "").strip(),
                    line[17:20].replace(" ", "").strip(),
                    line[21].replace(" ", "").strip(),
                    line[22:26].replace(" ", "").strip(),
                ]
            ]
            res_coords += [
                [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
            ]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    # Change the CA from GLY to CB, so it won't be excluded in the
    # atom_interaction_matrix
    res_data[:, 0][
        np.all(
            np.column_stack((res_data[:, 0] == "CA", res_data[:, 1] == "GLY")), axis=1
        )
    ] = "CB"

    # remove duplicated side chain entries and store only their first appearing
    rd_un, rd_uc = np.unique(res_data, axis=0, return_index=True)
    rd_uc = np.sort(rd_uc)
    res_data = res_data[rd_uc]
    res_coords = res_coords[rd_uc]
    return res_data, res_coords


def atom_interaction_matrix(
    path_to_pdb_file: str,
    plot_matrices: bool = False,
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """computes the adjacency matrix for a given pdb file based on the closest
    side chain atoms
    :parameter
        - path_to_pdb_file:
          path to pdb file of the protein of interest
        - plot_matrices:
          True to plot the distance matrix
    :returns
        adjacency is given per residue (the closest atom to any side chain atom of
        any other residue)
        - red2:
          adjacency (distance) matrix of the given protein with
          size len(protein_seq) x len(protein_seq)
    """
    # data [[ATOM, RES, CHAIN, ResNR],..]
    data, coords = data_coord_extraction(path_to_pdb_file)
    # ca alpha distances
    if plot_matrices:
        cab = data[:, 0] == "CA"
        dca = dist_calc(coords[cab], coords[cab])

    # to get only data and coords that belong to side chain atoms
    main_chain_label = np.invert(np.isin(data[:, 0], np.asarray(["C", "CA", "N", "O"])))
    data = data[main_chain_label]
    coords = coords[main_chain_label]

    # distance between all atoms
    d = dist_calc(coords, coords)

    # getting the start and end of each residues' entry in data
    udata, uind, ucount = np.unique(
        data[:, 1:], axis=0, return_index=True, return_counts=True
    )
    # sort it again by chain and sequence
    u_sort = np.lexsort((udata[:, 2].astype(int), udata[:, 1]))
    uind = uind[u_sort]
    ucount = ucount[u_sort]

    # reduce all distances to the closest distance of one side chain atom to another
    # per residue
    red1 = []
    for i, j in zip(uind, ucount):
        red1.append(np.min(d[:, i : i + j], axis=1))
    red1 = np.asarray(red1)

    red2 = []
    for i, j in zip(uind, ucount):
        red2.append(np.min(red1[:, i : i + j], axis=1))
    red2 = np.asarray(red2)

    if plot_matrices:
        fig, ax = plt.subplots(1, 1, figsize=(32, 18))
        ax.imshow(red2)
        plt.show()

    # one ResidueID per residue in the right order
    rd_un, rd_uc = np.unique(data[:, 1:], axis=0, return_index=True)
    rd_sort = np.sort(rd_uc)
    data = data[rd_sort][:, 1:]
    coords = coords[rd_sort]

    return data, coords, red2


def build_graph(
    data: np.ndarray[tuple[int, int], np.dtype[str]],
    coords: np.ndarray[tuple[int, int], np.dtype[float]],
    dist_matrix: np.ndarray[tuple[int, int], np.dtype[float]],
    dist_th: int | str = 5.0,
) -> tuple[
    list[str],
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
    dict,
]:
    """creates graph of interacting residues
    :parameter
        - pdb_filepath:
          filepath to the pdb file of the protein of interest
        - dist_th:
          maximum distance in \u212B of atoms of two residues to be seen as interacting
    :return
        - res_id_names
          ResidueIDs of all residues
        - reduced_coords
          coords of the residues that are closest to one other
        - dist_matrix
          n x n distance matrix between all residues
        - graph
          graph represented like
          {
          Residue1 : {Neighbour1: DistToResidue1, Neighbour2: DistToResidue1},
          ...
          }
    """
    # unique ResidueIDs as str
    res_id_names = np.asarray(["-".join(i) for i in data])
    # the residues as their index
    res_ids = np.arange(len(res_id_names))
    graph = {}
    # for all residues find residues that are closer than dist_th and ad them in a dict
    # to graph with the residue they are based on as key
    for i in res_ids:
        i_dists = dist_matrix[i]
        idist_bool = (i_dists <= dist_th) & (i_dists > 0.0)
        graph[i] = dict(zip(res_ids[idist_bool], i_dists[idist_bool]))
    return res_id_names, coords, dist_matrix, graph


if __name__ == "__main__":
    pass
