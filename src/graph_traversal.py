"""
graph_traversal.py

Provides utilities for traversing the
Knowledge Graph.
"""

import networkx as nx

from src.knowledge_graph import KnowledgeGraphBuilder


class GraphTraversal:

    def __init__(
            self,
            graph_path="data/knowledge_graph.pkl"
    ):

        self.graph = KnowledgeGraphBuilder.load_graph(
            graph_path
        )

    # ----------------------------------------------------
    # Search Node
    # ----------------------------------------------------

    def search_node(self, entity_name):
        """
        Search for a node (case-insensitive).
        """

        entity_name = entity_name.lower().strip()

        for node in self.graph.nodes():

            if node.lower() == entity_name:

                return node

        return None

    # ----------------------------------------------------
    # Get Outgoing Neighbors
    # ----------------------------------------------------

    def get_neighbors(self, entity_name):

        node = self.search_node(entity_name)

        if node is None:
            return []

        neighbors = []

        for neighbor in self.graph.neighbors(node):

            relation = self.graph.edges[
                node,
                neighbor
            ]["relation"]

            neighbors.append({

                "entity": neighbor,

                "relation": relation

            })

        return neighbors

    # ----------------------------------------------------
    # Get Incoming Neighbors
    # ----------------------------------------------------

    def get_incoming_neighbors(self, entity_name):

        node = self.search_node(entity_name)

        if node is None:
            return []

        incoming = []

        for predecessor in self.graph.predecessors(node):

            relation = self.graph.edges[
                predecessor,
                node
            ]["relation"]

            incoming.append({

                "entity": predecessor,

                "relation": relation

            })

        return incoming

    # ----------------------------------------------------
    # Get Relationships
    # ----------------------------------------------------

    def get_relationships(self, entity_name):

        return {

            "outgoing":

                self.get_neighbors(entity_name),

            "incoming":

                self.get_incoming_neighbors(entity_name)

        }

    # ----------------------------------------------------
    # Get Subgraph
    # ----------------------------------------------------

    def get_subgraph(
            self,
            entity_name,
            depth=1
    ):

        node = self.search_node(entity_name)

        if node is None:
            return nx.DiGraph()

        nodes = {node}

        frontier = {node}

        for _ in range(depth):

            next_frontier = set()

            for current in frontier:

                next_frontier.update(

                    self.graph.neighbors(current)

                )

                next_frontier.update(

                    self.graph.predecessors(current)

                )

            nodes.update(next_frontier)

            frontier = next_frontier

        return self.graph.subgraph(nodes).copy()

    # ----------------------------------------------------
    # Print Context
    # ----------------------------------------------------

    def print_context(self, entity_name):

        node = self.search_node(entity_name)

        if node is None:

            print("\nEntity not found.\n")

            return

        print()

        print("=" * 60)

        print("Topic :", node)

        print("=" * 60)

        print()

        print("Outgoing Relationships\n")

        for item in self.get_neighbors(node):

            print(

                f"{node}"

                f" --[{item['relation']}]--> "

                f"{item['entity']}"

            )

        print()

        print("Incoming Relationships\n")

        for item in self.get_incoming_neighbors(node):

            print(

                f"{item['entity']}"

                f" --[{item['relation']}]--> "

                f"{node}"

            )

        print()

    # ----------------------------------------------------
    # Get Context for Prompt Builder
    # ----------------------------------------------------

    def get_context(self, entity_name):
        """
        Returns graph context as a list of strings.
        """

        node = self.search_node(entity_name)

        if node is None:
            return []

        context = []

        # Outgoing

        for item in self.get_neighbors(node):

            context.append(

                f"{node} --[{item['relation']}]--> {item['entity']}"

            )

        # Incoming

        for item in self.get_incoming_neighbors(node):

            context.append(

                f"{item['entity']} --[{item['relation']}]--> {node}"

            )

        return context