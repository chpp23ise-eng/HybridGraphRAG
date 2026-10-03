import pickle

with open("data/knowledge_graph.pkl", "rb") as f:
    graph = pickle.load(f)

print(graph)

print("Nodes:", graph.number_of_nodes())

print("Edges:", graph.number_of_edges())

print("\nSample Relationships:\n")

for source, target, data in list(graph.edges(data=True))[:10]:
    print(f"{source} --[{data['relation']}]--> {target}")