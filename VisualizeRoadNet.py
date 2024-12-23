import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

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

# Visiting All Cities Exactly Once
def visiting_all_cities(roads, start):
    def visit(city, visited, current_cost):
        print(f"Visiting {city}, visited: {visited}, current cost: {current_cost}")
        if len(visited) == len(cities):
            for neighbor, distance in roads[city]:
                if neighbor == start:
                    return visited + [start], current_cost + distance
            return None, float('inf')  # If no path back to start city

        min_path, min_cost = None, float('inf')
        for neighbor, distance in roads.get(city, []):
            if neighbor not in visited:
                new_path, new_cost = visit(neighbor, visited + [neighbor], current_cost + distance)
                if new_cost < min_cost:
                    min_path, min_cost = new_path, new_cost
        return min_path, min_cost

    if start not in roads:
        return None, float('inf')

    return visit(start, [start], 0)

# Simple command-line interface for strategy selection
def main():
    start_city = 'Addis Ababa'
    goal_city = 'Mekelle'

    strategy = input("Choose the search strategy (bfs/dfs/ucs): ").strip().lower()

    if strategy not in ['bfs', 'dfs', 'ucs']:
        print("Invalid strategy selected. Please choose 'bfs', 'dfs', or 'ucs'.")
        return

    path, cost = uninformed_path_finder(cities, roads, start_city, goal_city, strategy)
    print(f"Path: {path} with cost {cost}")

    # Highlight the path found
    if path:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # Highlight start and goal cities
        nx.draw_networkx_nodes(G, pos, nodelist=[start_city], node_color='green', node_size=4000)
        nx.draw_networkx_nodes(G, pos, nodelist=[goal_city], node_color='red', node_size=4000)

        # Highlight the path found
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

        plt.show()

    # Example usage for Visiting All Cities Exactly Once
    visiting_path, visiting_cost = visiting_all_cities(roads, start_city)
    print(f"Visiting All Cities Path: {visiting_path} with cost {visiting_cost}")

    # Highlight the full path found for visiting all cities
    if visiting_path:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # Highlight start and return to start city
        nx.draw_networkx_nodes(G, pos, nodelist=[start_city], node_color='green', node_size=4000)

        # Highlight the full path found
        path_edges = [(visiting_path[i], visiting_path[i+1]) for i in range(len(visiting_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='purple', width=2)

        plt.show()

# Run the main function
if __name__ == "__main__":
    main()
