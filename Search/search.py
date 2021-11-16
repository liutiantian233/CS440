# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...

from collections import deque
import heapq


class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances = {
                (i, j): manhattan_distance(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    deck = deque()
    deck.append(maze.start)
    visited = set()
    route = {maze.start: None}
    path = deque()
    loop = True
    while loop:
        node = deck.popleft()
        visited.add(node)
        for neighbor in maze.neighbors(node[0], node[1]):
            if neighbor not in visited:
                deck.append(neighbor)
                visited.add(neighbor)
                route[neighbor] = node
                if neighbor == maze.waypoints[0]:
                    path.appendleft(neighbor)
                    backtracking = route[neighbor]
                    while backtracking in route:
                        path.appendleft(backtracking)
                        backtracking = route[backtracking]
                    loop = False
    return path


def manhattan_distance(coordinate_a, coordinate_b):
    return abs(coordinate_a[1] - coordinate_b[1]) + abs(coordinate_a[0] - coordinate_b[0])


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    distance = {maze.start: 0}
    deck = []
    heapq.heappush(deck, (distance[maze.start] + manhattan_distance(maze.start, maze.waypoints[0]), maze.start))
    visited = set()
    route = {maze.start: None}
    path = deque()
    loop = True
    while loop:
        node = heapq.heappop(deck)[1]
        visited.add(node)
        for neighbor in maze.neighbors(node[0], node[1]):
            if neighbor not in visited:
                distance[neighbor] = distance[node] + 1
                visited.add(neighbor)
                route[neighbor] = node
                heapq.heappush(deck, (distance[neighbor] + manhattan_distance(neighbor, maze.waypoints[0]), neighbor))
                if neighbor == maze.waypoints[0]:
                    path.appendleft(neighbor)
                    backtracking = route[neighbor]
                    while backtracking in route:
                        path.appendleft(backtracking)
                        backtracking = route[backtracking]
                    loop = False
    return path


def astar_or_fast(maze, weight):
    weight_dictionary = {}
    state = (maze.start, maze.waypoints)
    state_weight_dictionary = heuristic(state[0], state[1], weight_dictionary)
    distance = {state: 0}
    weight_dictionary = state_weight_dictionary[1]
    deck = []
    heapq.heappush(deck, (state_weight_dictionary[0] * weight, state))
    visited = set()
    route = {state: None}
    path = deque()
    loop = True
    while loop:
        node_state = heapq.heappop(deck)[1]
        node = node_state[0]
        node_visited = node_state[1]
        visited.add(node_state)
        if len(node_visited) == 1 and node in node_visited:
            path.appendleft(node_state[0])
            backtracking = route[node_state]
            while backtracking in route:
                path.appendleft(backtracking[0])
                backtracking = route[backtracking]
            break
        if node in node_visited:
            neighbor_visited = node_visited[:node_visited.index(node)] + node_visited[node_visited.index(node) + 1:]
        else:
            neighbor_visited = node_visited
        for neighbor in maze.neighbors(node[0], node[1]):
            neighbor_distance = distance[node_state] + 1
            if (neighbor, neighbor_visited) not in visited:
                visited.add((neighbor, neighbor_visited))
                distance[(neighbor, neighbor_visited)] = neighbor_distance
                next_weight = heuristic(neighbor, neighbor_visited, weight_dictionary)
                weight_dictionary = next_weight[1]
                heapq.heappush(deck, (neighbor_distance + next_weight[0] * weight, (neighbor, neighbor_visited)))
                route[(neighbor, neighbor_visited)] = (node, node_visited)
    return path


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_or_fast(maze, 1)


def heuristic(current_position, waypoints, weight_dictionary):
    distance = []
    for waypoint in waypoints:
        distance.append(manhattan_distance(current_position, waypoint))
    distance = min(distance)
    if waypoints in weight_dictionary:
        distance += weight_dictionary[waypoints]
    else:
        distance += MST(waypoints).compute_mst_weight()
        weight_dictionary[waypoints] = MST(waypoints).compute_mst_weight()
    return distance, weight_dictionary


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_or_fast(maze, 10)
