import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='Name of the TSP distance matrix CSV file')
parser.add_argument('--b', required=True)
args = parser.parse_args()

distance_matrix = pd.read_csv(f"./{args.filename}", index_col=0, na_values=['inf', 'Inf', 'INF'], keep_default_na=True)
distance_matrix.fillna(np.inf, inplace=True)

# Building the graph
cities = list(distance_matrix.columns)
n = len(cities)

G = nx.Graph()
for i in cities:
    for j in cities:
        if i != j:
            weight = distance_matrix.loc[i, j]
            if np.isfinite(weight):
                G.add_edge(i, j, weight=weight)

def tour_cost(tour, graph):
    """
    Calculates the total cost of a tour
    """
    cost = 0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if graph.has_edge(u, v):
            cost += graph[u][v]['weight']
        else:
            return float('inf')
    return cost

# Generate all possible tours
all_tours = list(itertools.permutations(cities))

# Finding the most optimal tour
minimal_tour = None
minimal_cost = float(args.b)

# ensure that the number of cities is greater than 2, else there will be no cycles possible
if n > 2:
    for tour in all_tours:
        current_cost = tour_cost(tour, G)
        if current_cost < minimal_cost:
            minimal_cost = current_cost
            minimal_tour = tour

# Setting up inputs for Reliability Network
if minimal_tour is None:
    print("No TSP tour exists.")
else:
    print(f"\nMinimal TSP Tour Cost (Budget b): {minimal_cost}")
    print(f"Minimal TSP Tour: {minimal_tour}")
    

# Budget for Reliability Network
b = float(args.b)

# Connectivity matrix with r_ij = 2 for all distinct pairs of nodes
requirement_matrix = pd.DataFrame(2, index=cities, columns=cities)
np.fill_diagonal(requirement_matrix.values, 0)

# Print the Connectivity Requirement Matrix (r_ij)
print("\nConnectivity Requirement Matrix (r_ij):")
print(requirement_matrix)

# Print the budget
print(f"\nBudget (b): {b}")

# Display the original graph G with edge weights
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

nx.draw_networkx_edges(G, pos)

nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

try:
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {edge: f"{weight:.1f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
except Exception as e:
    print(f"Error plotting edge labels: {e}")

plt.title('Original Graph with Edge Weights')
plt.axis('off')
plt.show()

# Reliabile network using an exponential search algorithm

# Get all possible edge subsets
all_edges = list(G.edges(data=True))

best_reliable_network = None
best_total_cost = float('inf')

found_solution = False


max_edges = len(all_edges)
for r in range(n - 1, max_edges + 1):  # At least n - 1 edges to be connected
    print(f"Checking combinations with {r} edges...")
    edge_combinations = combinations(all_edges, r)
    for edges_subset in edge_combinations:
        reliable_network = nx.Graph()
        reliable_network.add_nodes_from(G.nodes())
        total_cost = 0
        for u, v, data in edges_subset:
            weight = data['weight']
            reliable_network.add_edge(u, v, weight=weight)
            total_cost += weight
        if total_cost > b:
            continue 
        if nx.is_biconnected(reliable_network):
            all_requirements_met = True
            for i in cities:
                for j in cities:
                    if i != j:
                        num_paths = nx.node_connectivity(reliable_network, i, j)
                        if num_paths < requirement_matrix.loc[i, j]:
                            all_requirements_met = False
                            break
                if not all_requirements_met:
                    break
            if all_requirements_met:
                if total_cost < best_total_cost:
                    best_reliable_network = reliable_network.copy()
                    best_total_cost = total_cost
                found_solution = True
    if found_solution:
        break

if best_reliable_network is not None:
    print("\nReliable Network constructed successfully within the budget.")
    print(f"Total cost of the Reliable Network: {best_total_cost}")

    pos = nx.spring_layout(best_reliable_network, seed=42)

    nx.draw_networkx_nodes(best_reliable_network, pos, node_size=500, node_color='lightgreen')

    edge_labels = nx.get_edge_attributes(best_reliable_network, 'weight')
    nx.draw_networkx_edges(best_reliable_network, pos)
    nx.draw_networkx_edge_labels(best_reliable_network, pos, edge_labels=edge_labels)

    nx.draw_networkx_labels(best_reliable_network, pos, font_size=12, font_family='sans-serif')

    plt.title('Reliable Network Graph')
    plt.axis('off')
    plt.show()
else:
    print("\nCannot construct a Reliable Network within the budget that meets the connectivity requirements.")