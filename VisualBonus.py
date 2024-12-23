import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

from VisualizeRoadNet import visiting_all_cities

# Graph Creation and Visualization
G = nx.Graph()
cities = ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Hawassa', 'Mekelle']
G.add_nodes_from(cities)
roads = {
    'Addis Ababa': [('Bahir Dar', 510), ('Hawassa', 275)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300)],
    'Hawassa': [('Addis Ababa', 275)],
    'Mekelle': [('Gondar', 300)]
}
for city, connections in roads.items():
    for connected_city, distance in connections:
        G.add_edge(city, connected_city, weight=distance)

# Breadth-First Search (BFS)
def bfs(roads, start, goal):
    queue = deque([(start, [start], 0)])  # Add cumulative cost to queue
    visited = set()
    while queue:
        (vertex, path, cost) = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        for (neighbor, distance) in roads.get(vertex, []):
            if neighbor == goal:
                return path + [neighbor], cost + distance
            else:
                queue.append((neighbor, path + [neighbor], cost + distance))
    return None, float('inf')

# Depth-First Search (DFS)
def dfs(roads, start, goal):
    stack = [(start, [start], 0)]  # Add cumulative cost to stack
    visited = set()
    while stack:
        (vertex, path, cost) = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        for (neighbor, distance) in roads.get(vertex, []):
            if neighbor == goal:
                return path + [neighbor], cost + distance
            else:
                stack.append((neighbor, path + [neighbor], cost + distance))
    return None, float('inf')

# Uniform Cost Search (UCS)
def ucs(roads, start, goal):
    queue = [(0, start, [start])]  # cost, vertex, path
    visited = set()
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex == goal:
            return path, cost
        for (neighbor, distance) in roads.get(vertex, []):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + distance, neighbor, path + [neighbor]))
    return None, float('inf')
# Uninformed Path Finder
def uninformed_path_finder(cities, roads, start_city, goal_city, strategy):
    """
    Parameters:
    - cities: List of city names.
    - roads: Dictionary with city connections as {city: [(connected_city, distance)]}.
    - start_city: The city to start the journey.
    - goal_city: The destination city (for specific tasks).
    - strategy: The uninformed search strategy to use ('bfs', 'dfs', or 'ucs').

    Returns:
    - path: List of cities representing the path from start_city to goal_city.
    - cost: Total cost (number of steps or distance) of the path.
    """
    if strategy == 'bfs':
        return bfs(roads, start_city, goal_city)
    elif strategy == 'dfs':
        return dfs(roads, start_city, goal_city)
    elif strategy == 'ucs':
        return ucs(roads, start_city, goal_city)
    else:
        raise ValueError("Unsupported strategy. Use 'bfs', 'dfs', or 'ucs'.")

# Traverse All Cities
def traverse_all_cities(cities, roads, start_city, strategy):
    """
    Parameters:
    - cities: List of city names.
    - roads: Dictionary with city connections as {city: [(connected_city, distance)]}.
    - start_city: The city to start the journey.
    - strategy: The uninformed search strategy to use ('bfs' or 'dfs').

    Returns:
    - path: List of cities representing the traversal path.
    - cost: Total cost (distance) of the traversal.
    """
    if start_city not in cities:
        return None, float('inf')

    path = [start_city]
    total_cost = 0
    visited = set([start_city])
    current_city = start_city

    while len(visited) < len(cities):
        next_city = None
        min_cost = float('inf')

        for city in cities:
            if city != current_city and city not in visited:
                partial_path, cost = uninformed_path_finder(cities, roads, current_city, city, strategy)
                if cost < min_cost:
                    next_city = city
                    min_cost = cost

        if next_city is None:
            return None, float('inf')

        visited.add(next_city)
        path.append(next_city)
        total_cost += min_cost
        current_city = next_city

    return path, total_cost
# Dynamic Road Conditions (Bonus 1)
def handle_dynamic_conditions(roads, blocked_road):
    """
    Modify the roads dictionary to handle dynamic conditions.
    Parameters:
    - roads: Dictionary with city connections as {city: [(connected_city, distance)]}.
    - blocked_road: Tuple representing the blocked road as (city1, city2).

    Returns:
    - updated_roads: Modified roads dictionary.
    """
    city1, city2 = blocked_road
    updated_roads = {city: connections[:] for city, connections in roads.items()}  # Deep copy of roads

    if city1 in updated_roads:
        updated_roads[city1] = [c for c in updated_roads[city1] if c[0] != city2]

    if city2 in updated_roads:
        updated_roads[city2] = [c for c in updated_roads[city2] if c[0] != city1]

    return updated_roads

# K-Shortest Paths (Bonus 2)
def k_shortest_paths(cities, roads, start_city, goal_city, k):
    """
    Find the k-shortest paths between two cities.
    Parameters:
    - cities: List of city names.
    - roads: Dictionary with city connections as {city: [(connected_city, distance)]}.
    - start_city: The city to start the journey.
    - goal_city: The destination city.
    - k: Number of shortest paths to find.

    Returns:
    - paths: List of k paths, each represented as a tuple (path, cost).
    """
    def path_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            for neighbor, distance in roads[path[i]]:
                if neighbor == path[i + 1]:
                    cost += distance
                    break
        return cost

    # Use UCS to find paths
    queue = [(0, [start_city])]
    paths = []

    while queue and len(paths) < k:
        (cost, path) = heapq.heappop(queue)
        if path[-1] == goal_city:
            paths.append((path, cost))
        for (neighbor, distance) in roads.get(path[-1], []):
            if neighbor not in path:
                new_path = path + [neighbor]
                new_cost = path_cost(new_path)
                heapq.heappush(queue, (new_cost, new_path))

    return paths

# Simple command-line interface for strategy selection
def main():
    start_city = 'Addis Ababa'
    goal_city = 'Mekelle'

    strategy = input("Choose the search strategy (bfs/dfs/ucs): ").strip().lower()

    if strategy not in ['bfs', 'dfs', 'ucs']:
        print("Invalid strategy selected. Please choose 'bfs', 'dfs', or 'ucs'.")
        return

    # Traverse all cities
    traverse_path, traverse_cost = traverse_all_cities(cities, roads, start_city, strategy)
    print(f"Traverse All Cities Path: {traverse_path} with cost {traverse_cost}")

    # Handle dynamic road conditions
    blocked_road = ('Addis Ababa', 'Bahir Dar')
    updated_roads = handle_dynamic_conditions(roads, blocked_road)
    dynamic_path, dynamic_cost = traverse_all_cities(cities, updated_roads, start_city, strategy)
    print(f"Dynamic Conditions Path: {dynamic_path} with cost {dynamic_cost}")

    # Find k-shortest paths
    k = 3
    k_paths = k_shortest_paths(cities, roads, start_city, goal_city, k)
    for i, (path, cost) in enumerate(k_paths, 1):
        print(f"{i}-shortest path: {path} with cost {cost}")

    # Highlight the paths found
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    adjusted_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}  # Adjust position to avoid overlap
    nx.draw_networkx_edge_labels(G, adjusted_pos, edge_labels=labels)

    # Highlight start and goal cities
    nx.draw_networkx_nodes(G, pos, nodelist=[start_city], node_color='green', node_size=4000)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_city], node_color='red', node_size=4000)

    # Highlight the traversal path
    if traverse_path:
        path_edges = [(traverse_path[i], traverse_path[i+1]) for i in range(len(traverse_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2)

    # Highlight the dynamic conditions path
    if dynamic_path:
        dynamic_edges = [(dynamic_path[i], dynamic_path[i+1]) for i in range(len(dynamic_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=dynamic_edges, edge_color='orange', width=2)

    # Highlight the k-shortest paths
    for i, (path, _) in enumerate(k_paths):
        k_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=k_edges, edge_color='cyan', style='dashed', width=2)

    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
