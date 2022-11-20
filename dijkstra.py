from timeit import default_timer as timer
import numpy as np


def dijkstra(
    graph: dict, start_v: str | int, end_v: str | int, silent: bool = False
) -> tuple[list[str | int], list[str | int], list[str | int]]:
    """dijkstra's algorithm to find shortest path in graph
    :parameter
        - graph:
          graph represented like
          {
          Vertex1 : {Neighbour1: DistToVertex1, Neighbour2: DistToVertex1},
          ...
          }
        - start_v:
          name of the start vertex
        - end_v:
          name of the end vertex
        -silent:
          True to not print intermediate stats during runtime
    :return
        - vertex
          all vertices
        - dist_to_a
          all distances of vertices in vertex to start_time
        - prev_vertex
          parent vertex of all vertices
    """
    start_time = timer()

    # initialize lists of already visited and unvisited vertices
    visited = []
    unvisited = []

    # number of vertices in graph
    num_verteces = len(graph)

    # which vertices were already visited as boolean list
    visited_bool = np.ones(num_verteces).astype(bool)
    # vertex position in graph (dict)
    vertex_pos = {}
    # all vertices
    vertex = []
    # distance of each vertex to start
    dist_to_a = []
    # parent vertex of each vertex
    prev_vertex = []

    # initial fill of lists
    count = 0
    for key, value in graph.items():
        vertex_pos[key] = count
        dist = np.inf
        if key == start_v:
            dist = 0
        dist_to_a.append(dist)
        prev_vertex.append(None)
        unvisited.append(key)
        vertex.append(key)
        count += 1

    dist_to_a = np.asarray(dist_to_a)
    vertex = np.asarray(vertex)
    prev_vertex = np.asarray(prev_vertex)
    while len(unvisited) > 0:
        # vertex we currently looking at (closest to start)
        current_vertex = vertex[visited_bool][np.argmin(dist_to_a[visited_bool])]
        # its neighbours
        unvisited_neighbours = graph[current_vertex]
        # the so far closest distance to start
        smallest_dist = dist_to_a[vertex_pos[current_vertex]]
        # for all neighbours that were not already visited as vertices
        for key, value in unvisited_neighbours.items():
            if key not in visited:
                # distance of start to key vertex
                key_dist = value + smallest_dist
                # if new distance is closer update the closest dists to start and
                # over which parent vertex they were
                if key_dist < dist_to_a[vertex_pos[key]]:
                    dist_to_a[vertex_pos[key]] = key_dist
                    prev_vertex[vertex_pos[key]] = current_vertex
        # update lists
        visited.append(current_vertex)
        visited_bool[vertex_pos[current_vertex]] = False
        unvisited = vertex[visited_bool]
        # info
        if not silent:
            print(f"Current node: {current_vertex:>7}")
            print(f"Nodes visited: {len(visited):>6}")
            print(f"Nodes unvisited: {len(unvisited):>4}")
            print(f"time elapsed: {timer() - start_time:>7.5f}")
            print("-+-" * 7 + "\n")
    return vertex, dist_to_a, prev_vertex


def reconstruct_path(
    start_vertex: str | int,
    end_vertex: str | int,
    vert: list[str | int],
    dist: list[str | int],
    prev_vert: list[str | int],
) -> tuple[list[str | int], float]:
    """reconstruct shortest path found by the algorithm
    :parameter
        - start_vertex:
          name of the start vertex in the path
        - end_vertex:
          name of the end vertex in the path of interest
        - vert, dist, prev_vert:
          returns of the dijkstra function
    :return
        - visited_nodes:
          names of the nodes that construct the shortest path
        - end_vertex_dist:
          distance between start and end in the shortest path
    """
    if any(dist == np.inf):
        raise KeyError(
            f"Shortest path between selected vertices '{start_vertex}' "
            f"and '{end_vertex}' couldn't be constructed due to missing connections"
        )
    # list of nodes in the shortest path
    visited_nodes = [end_vertex]

    def iterative_add(cur_vertex: list[int | str]):
        """iteratively checks the parent of the current vertex and adds it to the
        visited_nodes
        :parameter
            - cur_vertex:
              name of the current vertex:
        :return
            - iterative_add
        """
        pos = np.where(vert == cur_vertex)[0]
        # parent of current node
        predecesor = prev_vert[pos]
        visited_nodes.append(predecesor[0])
        # stop when at start
        if predecesor != start_vertex:
            return iterative_add(predecesor)

    iterative_add(end_vertex)
    # distance between start and end
    end_vertex_dist = float(dist[np.where(vert == end_vertex)[0]])
    return visited_nodes, end_vertex_dist


if __name__ == "__main__":
    graph = {
        "A": {"B": 6, "D": 1},
        "B": {"A": 6, "D": 2, "E": 2, "C": 5},
        "C": {"B": 5, "E": 5},
        "D": {"A": 1, "B": 2, "E": 1},
        "E": {"D": 1, "B": 2, "C": 5},
    }

    start_v = "A"
    end_v = "C"
    # --------------------------------------------------------------------------------
    graph = {
        "S": {"A": 7, "B": 2, "C": 3},
        "A": {"S": 7, "B": 3, "D": 4},
        "B": {"S": 2, "A": 3, "D": 4, "H": 1},
        "C": {"S": 3, "L": 2},
        "D": {"A": 4, "B": 4, "F": 5},
        "H": {"B": 1, "F": 3, "G": 2},
        "G": {"H": 2, "E": 2},
        "F": {"D": 5, "H": 3},
        "L": {"C": 2, "I": 4, "J": 4},
        "I": {"L": 4, "J": 6, "K": 4},
        "J": {"L": 4, "I": 6, "K": 4},
        "K": {"I": 4, "J": 4, "E": 5},
        "E": {"G": 2, "K": 5},
    }
    start_v = "S"
    end_v = "E"
    # --------------------------------------------------------------------------------
    v, d, pv = dijkstra(graph, start_v, end_v)
    print(reconstruct_path(start_v, end_v, v, d, pv))
