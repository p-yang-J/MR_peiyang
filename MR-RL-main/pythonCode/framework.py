import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes with labels
nodes = {
    "PTA": "Prior Trajectory Acquisition",
    "DGC": "Damping Gradient Configuration",
    "BSFD": "Base Scaling Factor Definition",
    "ORL": "Offline Reinforcement Learning",
    "NRL": "Online Reinforcement Learning",
    "Y5I": "YOLOv5 Integration (Future Work)"
}
G.add_nodes_from(nodes)

# Add edges (arrows)
edges = [
    ("PTA", "DGC"),
    ("DGC", "BSFD"),
    ("BSFD", "ORL"),
    ("ORL", "NRL"),
    ("NRL", "Y5I"),
]
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))

nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9, with_labels=False)

nx.draw_networkx_labels(G, pos, labels=nodes)

plt.axis('off')
plt.show()
