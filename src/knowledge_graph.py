"""
knowledge_graph.py

Builds a focused Knowledge Graph using LLMGraphTransformer.

The graph construction process includes:

1. Controlled graph-extraction chunking
2. Controlled entity types
3. Controlled relationship types
4. Entity normalization
5. Duplicate entity merging
6. Generic/noise entity filtering
7. Duplicate relationship filtering
8. Self-loop removal
9. Configurable graph-size safeguards

The extraction strategy uses a bounded, representative subset
of each document so that local LLM inference remains practical.

The complete document remains available to ChromaDB. The reduced
input here is only for Knowledge Graph construction.
"""

import pickle
import re
import networkx as nx
from pathlib import Path

from langchain_core.documents import Document

from langchain_experimental.graph_transformers import (
    LLMGraphTransformer
)


class KnowledgeGraphBuilder:

    # ========================================================
    # GRAPH SIZE SETTINGS
    # ========================================================

    MAX_NODES = 300

    MAX_RELATIONSHIPS = 500

    # ========================================================
    # GRAPH EXTRACTION SETTINGS
    # ========================================================

    # Number of words sent to the LLM in one graph extraction
    # request.
    #
    # Smaller chunks reduce the size of the prompt sent to
    # local Mistral and make each Ollama request faster.
    GRAPH_CHUNK_WORDS = 750

    # No overlap is used.
    #
    # Overlap causes the same concepts to be extracted
    # repeatedly and increases LLM calls.
    GRAPH_CHUNK_OVERLAP = 0

    # Maximum total number of words used from ONE document
    # for Knowledge Graph extraction.
    #
    # IMPORTANT:
    # This does NOT modify the original document or ChromaDB.
    #
    # If a document is larger than this value, representative
    # regions are selected across the entire document.
    GRAPH_MAX_WORDS_PER_DOCUMENT = 3000

    # ========================================================
    # ALLOWED NODE TYPES
    # ========================================================

    ALLOWED_NODE_TYPES = [

        "Concept",
        "Topic",
        "Event",
        "Process",
        "Method",
        "Technology",
        "Technique",
        "Material",
        "Product",
        "Organization",
        "Institution",
        "Person",
        "Location",
        "Country",
        "Region",
        "Sector",
        "Industry",
        "System",
        "Theory",
        "Model",
        "Problem",
        "Challenge",
        "Issue",
        "Outcome",
        "Impact",
        "Result",
        "Practice",
        "Resource",
        "Crop",
        "Agricultural Practice",
        "Disease",
        "Chemical",
        "Policy",
        "Program",
        "Metric",
        "Dataset"

    ]

    # ========================================================
    # ALLOWED RELATIONSHIPS
    # ========================================================

    ALLOWED_RELATIONSHIPS = [

        "PART_OF",
        "HAS_PART",
        "INCLUDES",
        "CONTAINS",

        "CAUSES",
        "LEADS_TO",
        "RESULTS_IN",

        "IMPACTS",
        "IMPACTED",
        "AFFECTS",
        "AFFECTED_BY",

        "INCREASES",
        "DECREASES",
        "INCREASED_BY",
        "DECREASED_BY",

        "USED_IN",
        "USED_FOR",

        "INTRODUCED",
        "FOLLOWED_BY",
        "PRECEDED_BY",

        "OCCURRED_IN",
        "OCCURRED",

        "APPLIED_IN",

        "REQUIRES",
        "DEPENDS_ON",

        "CONTRIBUTES_TO",
        "INFLUENCES",

        "PRODUCES",
        "PRODUCED_BY",

        "AT_EXPENSE_OF",
        "REPLACED_BY"

    ]

    # ========================================================
    # GENERIC / NOISE ENTITIES
    # ========================================================

    GENERIC_ENTITIES = {

        "thing",
        "things",

        "aspect",
        "aspects",

        "factor",
        "factors",

        "issue",
        "issues",

        "problem",
        "problems",

        "change",
        "changes",

        "development",
        "developments",

        "growth",

        "increase",
        "decrease",

        "result",
        "results",

        "effect",
        "effects",

        "impact",
        "impacts",

        "process",
        "processes",

        "method",
        "methods",

        "approach",
        "approaches",

        "system",
        "systems",

        "level",
        "levels",

        "type",
        "types",

        "example",
        "examples",

        "information",

        "data",

        "value",
        "values",

        "period",
        "periods",

        "time",
        "times",

        "way",
        "ways",

        "part",
        "parts",

        "area",
        "areas",

        "field",
        "fields",

        "study",
        "studies",

        "case",
        "cases",

        "section",
        "sections",

        "chapter",
        "chapters",

        "role",
        "roles",

        "use",
        "uses",

        "application",
        "applications",

        "technology",
        "technologies",

        "practice",
        "practices"

    }

    # ========================================================
    # GENERIC RELATIONSHIPS
    # ========================================================

    GENERIC_RELATIONSHIPS = {

        "",

        "UNKNOWN",

        "NONE",

        "RELATED",

        "RELATED_TO",

        "CONNECTED_TO",

        "ASSOCIATED"

    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, llm):

        self.llm = llm

        self.transformer = LLMGraphTransformer(

            llm=self.llm,

            allowed_nodes=self.ALLOWED_NODE_TYPES,

            allowed_relationships=self.ALLOWED_RELATIONSHIPS,

            strict_mode=True

        )

        self.graph = nx.DiGraph()

        # Normalized entity key -> canonical entity name
        self.entity_registry = {}

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    def build_graph(
        self,
        documents
    ):

        print(
            "\nBuilding Knowledge Graph...\n"
        )

        total_graph_chunks = 0
        total_graph_words = 0

        # ----------------------------------------------------
        # Process each source document
        # ----------------------------------------------------

        for document_index, document in enumerate(
            documents
        ):

            source = (
                document.metadata.get(
                    "source",
                    ""
                )
                if hasattr(
                    document,
                    "metadata"
                )
                else ""
            )

            source_name = Path(
                str(source)
            ).name if source else (
                f"Document {document_index + 1}"
            )

            # ------------------------------------------------
            # Prepare bounded KG extraction chunks
            # ------------------------------------------------

            graph_chunks = (
                self._split_document_for_graph(
                    document
                )
            )

            total_graph_chunks += len(
                graph_chunks
            )

            document_graph_words = sum(
                len(
                    chunk.page_content.split()
                )
                for chunk in graph_chunks
            )

            total_graph_words += (
                document_graph_words
            )

            print(
                f"Document "
                f"{document_index + 1}/"
                f"{len(documents)}: "
                f"{source_name}"
            )

            print(
                f"  Original words: "
                f"{len(document.page_content.split())}"
            )

            print(
                f"  KG input words: "
                f"{document_graph_words}"
            )

            print(
                f"  Graph extraction chunks: "
                f"{len(graph_chunks)}"
            )

            # ------------------------------------------------
            # LLM extraction
            # ------------------------------------------------

            for chunk_index, chunk in enumerate(
                graph_chunks
            ):

                print(
                    f"  Processing graph chunk "
                    f"{chunk_index + 1}/"
                    f"{len(graph_chunks)}..."
                )

                try:

                    result = (
                        self.transformer
                        .convert_to_graph_documents(
                            [chunk]
                        )
                    )

                    self._add_graph_documents(
                        result
                    )

                    print(
                        f"  Finished graph chunk "
                        f"{chunk_index + 1}"
                    )

                except Exception as e:

                    print(
                        f"  Skipping graph chunk "
                        f"{chunk_index + 1}: "
                        f"{e}"
                    )

        # ----------------------------------------------------
        # Remove isolated nodes
        # ----------------------------------------------------

        self._remove_isolated_nodes()

        # ----------------------------------------------------
        # Enforce graph size
        # ----------------------------------------------------

        self._limit_nodes()

        self._limit_relationships()

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        self._remove_duplicate_semantic_edges()

        self._remove_isolated_nodes()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print()

        print(
            "=" * 60
        )

        print(
            "Knowledge Graph Construction Complete"
        )

        print(
            "=" * 60
        )

        print(
            f"Documents : "
            f"{len(documents)}"
        )

        print(
            f"Graph Chunks : "
            f"{total_graph_chunks}"
        )

        print(
            f"KG Input Words : "
            f"{total_graph_words}"
        )

        print(
            f"Nodes : "
            f"{self.graph.number_of_nodes()}"
        )

        print(
            f"Relationships : "
            f"{self.graph.number_of_edges()}"
        )

        print()

        return self.graph

    # ========================================================
    # SPLIT DOCUMENT FOR GRAPH EXTRACTION
    # ========================================================

    def _split_document_for_graph(
        self,
        document
    ):
        """
        Prepares bounded, representative graph-extraction
        chunks from a document.

        The complete document is still preserved for ChromaDB.

        For documents within GRAPH_MAX_WORDS_PER_DOCUMENT,
        the complete text is chunked normally.

        For larger documents, representative windows are
        distributed across the document so that the beginning,
        middle, and later sections can all contribute entities
        and relationships to the Knowledge Graph.

        No remaining text is appended to the final chunk.
        """

        text = str(
            getattr(
                document,
                "page_content",
                ""
            )
        ).strip()

        if not text:

            return []

        words = text.split()

        # ----------------------------------------------------
        # Small document
        # ----------------------------------------------------

        if len(words) <= self.GRAPH_MAX_WORDS_PER_DOCUMENT:

            return self._create_graph_chunks(
                document=document,
                words=words,
                start_positions=range(
                    0,
                    len(words),
                    self.GRAPH_CHUNK_WORDS
                )
            )

        # ----------------------------------------------------
        # Large document
        #
        # Select representative regions across the complete
        # document instead of taking only the beginning.
        # ----------------------------------------------------

        target_words = (
            self.GRAPH_MAX_WORDS_PER_DOCUMENT
        )

        number_of_chunks = (
            target_words
            // self.GRAPH_CHUNK_WORDS
        )

        if number_of_chunks <= 0:

            number_of_chunks = 1

        # ----------------------------------------------------
        # Spread chunk starting positions across the document.
        #
        # Example:
        #
        # 14,637-word document
        # 4 x 750-word regions
        #
        # The regions are distributed from the beginning
        # towards the end rather than concentrated at the top.
        # ----------------------------------------------------

        max_start = max(
            0,
            len(words) - self.GRAPH_CHUNK_WORDS
        )

        if number_of_chunks == 1:

            start_positions = [0]

        else:

            start_positions = []

            for index in range(
                number_of_chunks
            ):

                position = round(
                    index
                    * max_start
                    / (
                        number_of_chunks - 1
                    )
                )

                start_positions.append(
                    position
                )

        return self._create_graph_chunks(
            document=document,
            words=words,
            start_positions=start_positions
        )

    # ========================================================
    # CREATE GRAPH CHUNKS
    # ========================================================

    def _create_graph_chunks(
        self,
        document,
        words,
        start_positions
    ):
        """
        Creates bounded Document objects from selected word
        positions.

        This method deliberately does not append leftover text
        to the final chunk.
        """

        chunks = []

        total_words = len(words)

        for chunk_number, start in enumerate(
            start_positions,
            start=1
        ):

            start = max(
                0,
                min(
                    int(start),
                    max(
                        0,
                        total_words - 1
                    )
                )
            )

            end = min(
                start + self.GRAPH_CHUNK_WORDS,
                total_words
            )

            chunk_text = " ".join(
                words[start:end]
            ).strip()

            if not chunk_text:
                continue

            metadata = dict(
                getattr(
                    document,
                    "metadata",
                    {}
                )
                or {}
            )

            metadata[
                "graph_chunk"
            ] = chunk_number

            metadata[
                "graph_chunk_start"
            ] = start

            metadata[
                "graph_chunk_end"
            ] = end

            metadata[
                "graph_chunk_source_words"
            ] = total_words

            chunks.append(

                Document(

                    page_content=chunk_text,

                    metadata=metadata

                )

            )

        return chunks

    # ========================================================
    # ADD GRAPH DOCUMENTS
    # ========================================================

    def _add_graph_documents(
        self,
        graph_documents
    ):

        for graph_document in graph_documents:

            # -----------------------------------------------
            # Nodes
            # -----------------------------------------------

            for node in graph_document.nodes:

                original_name = str(
                    node.id
                ).strip()

                if not original_name:
                    continue

                canonical_name = (
                    self._register_entity(
                        original_name
                    )
                )

                if canonical_name is None:
                    continue

                node_type = (
                    self._clean_node_type(
                        getattr(
                            node,
                            "type",
                            "Entity"
                        )
                    )
                )

                self.graph.add_node(

                    canonical_name,

                    type=node_type

                )

            # -----------------------------------------------
            # Relationships
            # -----------------------------------------------

            for relationship in (
                graph_document.relationships
            ):

                source_name = str(
                    relationship.source.id
                ).strip()

                target_name = str(
                    relationship.target.id
                ).strip()

                if (
                    not source_name
                    or not target_name
                ):

                    continue

                source = (
                    self._canonical_name(
                        source_name
                    )
                )

                target = (
                    self._canonical_name(
                        target_name
                    )
                )

                if (
                    source is None
                    or target is None
                ):

                    continue

                relation = (
                    self._clean_relationship(
                        relationship.type
                    )
                )

                if relation is None:
                    continue

                # -------------------------------------------
                # Remove self relationships
                # -------------------------------------------

                if (
                    self._normalize_key(source)
                    ==
                    self._normalize_key(target)
                ):

                    continue

                # -------------------------------------------
                # Add / update relationship
                # -------------------------------------------

                if self.graph.has_edge(
                    source,
                    target
                ):

                    existing_relation = (
                        self.graph.edges[
                            source,
                            target
                        ].get(
                            "relation",
                            ""
                        )
                    )

                    # Keep the existing relationship when
                    # the same source-target pair is already
                    # represented.
                    if existing_relation:
                        continue

                self.graph.add_edge(

                    source,

                    target,

                    relation=relation

                )

    # ========================================================
    # ENTITY REGISTRATION
    # ========================================================

    def _register_entity(
        self,
        entity_name
    ):

        cleaned = (
            self._clean_entity_name(
                entity_name
            )
        )

        if cleaned is None:
            return None

        key = self._normalize_key(
            cleaned
        )

        if key in self.entity_registry:

            return self.entity_registry[key]

        self.entity_registry[key] = cleaned

        return cleaned

    # ========================================================
    # ENTITY CANONICALIZATION
    # ========================================================

    def _canonical_name(
        self,
        entity_name
    ):

        cleaned = (
            self._clean_entity_name(
                entity_name
            )
        )

        if cleaned is None:
            return None

        key = self._normalize_key(
            cleaned
        )

        if key in self.entity_registry:

            return self.entity_registry[key]

        self.entity_registry[key] = cleaned

        return cleaned

    # ========================================================
    # ENTITY CLEANING
    # ========================================================

    def _clean_entity_name(
        self,
        entity_name
    ):

        if entity_name is None:
            return None

        text = str(
            entity_name
        ).strip()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip(
            " \t\n\r.,;:!?()[]{}\"'"
        )

        if not text:
            return None

        # ----------------------------------------------------
        # Prevent paragraph-like entities
        # ----------------------------------------------------

        if len(text) > 150:
            return None

        # ----------------------------------------------------
        # Ignore pure numbers
        # ----------------------------------------------------

        if re.fullmatch(
            r"[\d\s.,%-]+",
            text
        ):

            return None

        normalized = (
            self._normalize_key(
                text
            )
        )

        # ----------------------------------------------------
        # Generic entity filtering
        # ----------------------------------------------------

        if normalized in self.GENERIC_ENTITIES:
            return None

        # ----------------------------------------------------
        # Extremely short strings
        # ----------------------------------------------------

        if len(normalized) < 2:
            return None

        return text

    # ========================================================
    # NODE TYPE CLEANING
    # ========================================================

    def _clean_node_type(
        self,
        node_type
    ):

        if node_type is None:
            return "Entity"

        text = str(
            node_type
        ).strip()

        if not text:
            return "Entity"

        for allowed in self.ALLOWED_NODE_TYPES:

            if (
                allowed.lower()
                ==
                text.lower()
            ):

                return allowed

        return "Entity"

    # ========================================================
    # RELATIONSHIP CLEANING
    # ========================================================

    def _clean_relationship(
        self,
        relationship
    ):

        if relationship is None:
            return None

        relation = str(
            relationship
        ).strip()

        if not relation:
            return None

        relation = re.sub(
            r"[\s\-]+",
            "_",
            relation
        )

        relation = re.sub(
            r"_+",
            "_",
            relation
        )

        relation = relation.upper().strip(
            "_"
        )

        # ----------------------------------------------------
        # Remove vague relationships
        # ----------------------------------------------------

        if relation in self.GENERIC_RELATIONSHIPS:
            return None

        # ----------------------------------------------------
        # Common aliases
        # ----------------------------------------------------

        aliases = {

            "IMPACT":
                "IMPACTS",

            "AFFECT":
                "AFFECTS",

            "LEAD_TO":
                "LEADS_TO",

            "RESULT_IN":
                "RESULTS_IN",

            "OCCURS_IN":
                "OCCURRED_IN",

            "FOLLOW":
                "FOLLOWED_BY",

            "ASSOCIATED":
                "ASSOCIATED_WITH",

            "PART":
                "PART_OF",

            "INCLUDE":
                "INCLUDES",

            "USE":
                "USED_IN"

        }

        relation = aliases.get(
            relation,
            relation
        )

        # ----------------------------------------------------
        # Keep only controlled relationships
        # ----------------------------------------------------

        if (
            relation
            not in self.ALLOWED_RELATIONSHIPS
        ):

            return None

        return relation

    # ========================================================
    # NORMALIZATION KEY
    # ========================================================

    @staticmethod
    def _normalize_key(
        text
    ):

        text = str(
            text
        ).strip().lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # REMOVE ISOLATED NODES
    # ========================================================

    def _remove_isolated_nodes(self):

        isolated_nodes = [

            node

            for node in self.graph.nodes()

            if self.graph.degree(node) == 0

        ]

        self.graph.remove_nodes_from(
            isolated_nodes
        )

    # ========================================================
    # NODE LIMIT
    # ========================================================

    def _limit_nodes(self):

        if (
            self.graph.number_of_nodes()
            <= self.MAX_NODES
        ):

            return

        print()

        print(
            f"Node limit reached: "
            f"{self.MAX_NODES}"
        )

        ranked_nodes = sorted(

            self.graph.degree(),

            key=lambda item:
                item[1],

            reverse=True

        )

        keep_nodes = {

            node

            for node, _degree
            in ranked_nodes[
                :self.MAX_NODES
            ]

        }

        remove_nodes = [

            node

            for node
            in self.graph.nodes()

            if node not in keep_nodes

        ]

        self.graph.remove_nodes_from(
            remove_nodes
        )

    # ========================================================
    # RELATIONSHIP LIMIT
    # ========================================================

    def _limit_relationships(self):

        if (
            self.graph.number_of_edges()
            <= self.MAX_RELATIONSHIPS
        ):

            return

        print()

        print(
            f"Relationship limit reached: "
            f"{self.MAX_RELATIONSHIPS}"
        )

        ranked_edges = []

        for source, target, data in (
            self.graph.edges(
                data=True
            )
        ):

            score = (
                self.graph.degree(source)
                +
                self.graph.degree(target)
            )

            ranked_edges.append(
                (
                    score,
                    source,
                    target,
                    data
                )
            )

        ranked_edges.sort(
            reverse=True,
            key=lambda item:
                item[0]
        )

        keep_edges = ranked_edges[
            :self.MAX_RELATIONSHIPS
        ]

        keep_edge_keys = {

            (
                source,
                target
            )

            for _score,
            source,
            target,
            _data
            in keep_edges

        }

        remove_edges = [

            (
                source,
                target
            )

            for source, target
            in self.graph.edges()

            if (
                source,
                target
            ) not in keep_edge_keys

        ]

        self.graph.remove_edges_from(
            remove_edges
        )

        self._remove_isolated_nodes()

    # ========================================================
    # REMOVE DUPLICATE SEMANTIC EDGES
    # ========================================================

    def _remove_duplicate_semantic_edges(self):

        edges_to_remove = []

        seen = set()

        for source, target, data in (
            self.graph.edges(
                data=True
            )
        ):

            source_key = (
                self._normalize_key(
                    source
                )
            )

            target_key = (
                self._normalize_key(
                    target
                )
            )

            relation_key = (
                self._normalize_key(
                    data.get(
                        "relation",
                        ""
                    )
                )
            )

            key = (
                source_key,
                target_key,
                relation_key
            )

            if key in seen:

                edges_to_remove.append(
                    (
                        source,
                        target
                    )
                )

            else:

                seen.add(
                    key
                )

        self.graph.remove_edges_from(
            edges_to_remove
        )

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    def save_graph(
        self,
        path="data/knowledge_graph.pkl"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "wb"
        ) as file:

            pickle.dump(
                self.graph,
                file
            )

        print(
            f"\nKnowledge Graph saved to "
            f"{path}"
        )

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    @staticmethod
    def load_graph(
        path="data/knowledge_graph.pkl"
    ):

        with open(
            path,
            "rb"
        ) as file:

            graph = pickle.load(
                file
            )

        return graph

    # ========================================================
    # STATISTICS
    # ========================================================

    def print_statistics(self):

        print()

        print(
            "=" * 50
        )

        print(
            "Knowledge Graph Statistics"
        )

        print(
            "=" * 50
        )

        print()

        print(
            "Nodes :",
            self.graph.number_of_nodes()
        )

        print(
            "Edges :",
            self.graph.number_of_edges()
        )

        if self.graph.number_of_nodes() > 0:

            degrees = [

                degree

                for _node, degree
                in self.graph.degree()

            ]

            print(
                "Average Connections :",
                round(
                    sum(degrees)
                    /
                    len(degrees),
                    2
                )
            )

            print(
                "Maximum Connections :",
                max(degrees)
            )

        print()

    # ========================================================
    # SAMPLE RELATIONSHIPS
    # ========================================================

    def print_sample_relationships(
        self,
        limit=10
    ):

        print()

        print(
            "=" * 50
        )

        print(
            "Sample Relationships"
        )

        print(
            "=" * 50
        )

        print()

        count = 0

        for source, target, data in (
            self.graph.edges(
                data=True
            )
        ):

            print(

                f"{source}"

                f" --[{data['relation']}]--> "

                f"{target}"

            )

            count += 1

            if count >= limit:
                break

        print()