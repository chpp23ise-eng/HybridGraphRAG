from src.graph_traversal import GraphTraversal

from src.prompt_builder import PromptBuilder

graph = GraphTraversal()

topic = input("Enter Topic: ")

neighbors = graph.get_neighbors(topic)

prompt = PromptBuilder.build_prompt(

    topic,

    neighbors,

    mcq=5,

    short_questions=3,

    long_questions=2,

    difficulty="Medium"

)

print()

print(prompt)