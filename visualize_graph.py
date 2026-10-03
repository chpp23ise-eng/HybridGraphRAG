import pickle
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Load Knowledge Graph
# -----------------------------

with open("data/knowledge_graph.pkl", "rb") as file:
    graph = pickle.load(file)

print(f"Nodes : {graph.number_of_nodes()}")
print(f"Edges : {graph.number_of_edges()}")

# -----------------------------
# Draw Graph
# -----------------------------

plt.figure(figsize=(16, 12))

pos = nx.spring_layout(
    graph,
    seed=42,
    k=1.2
)

nx.draw_networkx_nodes(
    graph,
    pos,
    node_size=1800,
    alpha=0.9
)

nx.draw_networkx_edges(
    graph,
    pos,
    arrows=True,
    arrowsize=20,
    edge_color="gray"
)

nx.draw_networkx_labels(
    graph,
    pos,
    font_size=9,
    font_weight="bold"
)

edge_labels = nx.get_edge_attributes(graph, "relation")

nx.draw_networkx_edge_labels(
    graph,
    pos,
    edge_labels=edge_labels,
    font_size=8
)

plt.title("Knowledge Graph", fontsize=18)

plt.axis("off")

plt.tight_layout()

plt.show()