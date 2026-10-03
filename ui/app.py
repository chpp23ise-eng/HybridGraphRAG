"""
ui/app.py

Streamlit frontend for Hybrid GraphRAG.

Flow:

1. Show the complete Knowledge Graph before a topic is entered.
2. Once a topic is entered, show its local Knowledge Graph.
3. Configure the question paper pattern.
4. Select question types:
   - MCQ
   - Fill in the Blanks
   - Short Answer
   - Long Answer
5. Select Bloom's Taxonomy levels for every section.
6. Generate the Question Paper through FastAPI.
7. Display the generated Question Paper.
8. Allow PDF download.
"""

# ============================================================
# PROJECT PATH
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import requests
import streamlit.components.v1 as components
import networkx as nx

from utils.pdf_generator import PDFGenerator
from src.knowledge_graph import KnowledgeGraphBuilder
from src.graph_traversal import GraphTraversal


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/generate-paper"
)

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "knowledge_graph.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid GraphRAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 Hybrid GraphRAG Question Paper Generator"
)

st.write(
    "Generate question papers using "
    "Hybrid RAG + Knowledge Graph + "
    "Bloom's Taxonomy."
)

st.divider()


# ============================================================
# CONSTANTS
# ============================================================

QUESTION_TYPES = [

    "MCQ",

    "Fill in the Blanks",

    "Short Answer",

    "Long Answer"

]


BLOOM_LEVELS = [

    "L1 - Remember",

    "L2 - Understand",

    "L3 - Apply",

    "L4 - Analyze",

    "L5 - Evaluate",

    "L6 - Create"

]


# ============================================================
# KNOWLEDGE GRAPH DISPLAY
# ============================================================

def display_knowledge_graph(
    topic=None,
    depth=2
):

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    if topic and topic.strip():

        st.subheader(
            "🕸️ Topic Knowledge Graph"
        )

    else:

        st.subheader(
            "🕸️ Knowledge Graph"
        )

    # --------------------------------------------------------
    # Check Graph
    # --------------------------------------------------------

    if not os.path.exists(
        GRAPH_PATH
    ):

        st.warning(
            "Knowledge Graph not found."
        )

        return

    try:

        # ----------------------------------------------------
        # Load Complete Graph
        # ----------------------------------------------------

        graph = (
            KnowledgeGraphBuilder
            .load_graph(
                GRAPH_PATH
            )
        )

        if graph.number_of_nodes() == 0:

            st.warning(
                "Knowledge Graph is empty."
            )

            return

        # ----------------------------------------------------
        # Find Topic Node
        # ----------------------------------------------------

        selected_node = None

        if topic and topic.strip():

            traversal = GraphTraversal(
                GRAPH_PATH
            )

            selected_node = (
                traversal.search_node(
                    topic.strip()
                )
            )

        # ----------------------------------------------------
        # Build Display Graph
        #
        # IMPORTANT:
        # We calculate the local neighborhood directly here
        # instead of relying on the visual structure returned
        # by GraphTraversal.
        # ----------------------------------------------------

        if (
            topic
            and topic.strip()
            and selected_node is not None
            and selected_node in graph
        ):

            undirected_graph = (
                graph.to_undirected()
            )

            distances = (
                nx.single_source_shortest_path_length(
                    undirected_graph,
                    selected_node,
                    cutoff=depth
                )
            )

            visible_nodes = set(
                distances.keys()
            )

            display_graph = (
                graph.subgraph(
                    visible_nodes
                ).copy()
            )

            st.info(
                f"Showing **{selected_node}** "
                f"and entities within "
                f"**{depth} relationship level(s)**."
            )

        elif (
            topic
            and topic.strip()
            and selected_node is None
        ):

            st.warning(
                f"No entity matching "
                f"'{topic}' was found in the "
                f"Knowledge Graph."
            )

            display_graph = graph

        else:

            display_graph = graph

        # ----------------------------------------------------
        # Graph Statistics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Nodes",
                display_graph.number_of_nodes()
            )

        with col2:

            st.metric(
                "Relationships",
                display_graph.number_of_edges()
            )

        with col3:

            if topic and topic.strip():

                st.metric(
                    "Traversal Depth",
                    depth
                )

            else:

                st.metric(
                    "Graph",
                    "Complete"
                )

        # ----------------------------------------------------
        # Graph Description
        # ----------------------------------------------------

        if (
            topic
            and topic.strip()
            and selected_node is not None
        ):

            st.caption(
                "The selected topic is placed at the center. "
                "Surrounding nodes represent entities connected "
                "to the topic through the selected traversal depth. "
                "Arrows show the direction of extracted relationships."
            )

        else:

            st.caption(
                "The complete Knowledge Graph is shown as "
                "separate connected components. Each edge "
                "represents a relationship extracted from "
                "the source material."
            )

        # ====================================================
        # PYVIS
        # ====================================================

        from pyvis.network import Network

        net = Network(

            height="700px",

            width="100%",

            directed=True,

            bgcolor="#ffffff",

            font_color="#0f172a",

            notebook=False

        )

        # ----------------------------------------------------
        # Helper: Readable Node Labels
        # ----------------------------------------------------

        def clean_label(
            text,
            max_length=32
        ):

            text = str(text).replace(
                "_",
                " "
            )

            text = " ".join(
                text.split()
            )

            if len(text) <= max_length:

                return text

            return (
                text[:max_length - 3]
                + "..."
            )

        # ----------------------------------------------------
        # Helper: Wrap Text
        # ----------------------------------------------------

        def wrap_text(
            text,
            max_length=20
        ):

            words = str(text).split()

            lines = []

            current_line = ""

            for word in words:

                if not current_line:

                    current_line = word

                elif (
                    len(current_line)
                    + len(word)
                    + 1
                    <= max_length
                ):

                    current_line += (
                        " "
                        + word
                    )

                else:

                    lines.append(
                        current_line
                    )

                    current_line = word

            if current_line:

                lines.append(
                    current_line
                )

            return "\n".join(
                lines
            )

        # ----------------------------------------------------
        # Determine Center Node
        # ----------------------------------------------------

        center_node = None

        if (
            topic
            and topic.strip()
            and selected_node in display_graph
        ):

            center_node = selected_node

        elif display_graph.number_of_nodes() > 0:

            try:

                center_node = max(
                    display_graph.nodes(),
                    key=lambda node:
                        display_graph.degree(node)
                )

            except Exception:

                center_node = list(
                    display_graph.nodes()
                )[0]

        # ----------------------------------------------------
        # Calculate Distances From Center
        # ----------------------------------------------------

        node_levels = {}

        if center_node is not None:

            try:

                undirected_graph = (
                    display_graph
                    .to_undirected()
                )

                node_levels = (
                    nx.single_source_shortest_path_length(
                        undirected_graph,
                        center_node
                    )
                )

            except Exception:

                node_levels = {}

        # ====================================================
        # NODE POSITIONS
        #
        # Topic graph:
        #   Center node at the middle
        #   Level 1 on first ring
        #   Level 2 on second ring
        #
        # Complete graph:
        #   Components arranged independently.
        # ====================================================

        positions = {}

        if (
            topic
            and topic.strip()
            and center_node is not None
        ):

            # ------------------------------------------------
            # Group nodes by level
            # ------------------------------------------------

            levels = {}

            for node, level in node_levels.items():

                levels.setdefault(
                    level,
                    []
                ).append(node)

            # ------------------------------------------------
            # Center
            # ------------------------------------------------

            positions[center_node] = (
                0,
                0
            )

            # ------------------------------------------------
            # Concentric rings
            # ------------------------------------------------

            for level in sorted(
                levels.keys()
            ):

                if level == 0:

                    continue

                nodes_at_level = levels[
                    level
                ]

                radius = (
                    260 * level
                )

                count = len(
                    nodes_at_level
                )

                for index, node in enumerate(
                    nodes_at_level
                ):

                    angle = (
                        2
                        * 3.141592653589793
                        * index
                        / max(count, 1)
                    )

                    x = (
                        radius
                        * __import__(
                            "math"
                        ).cos(angle)
                    )

                    y = (
                        radius
                        * __import__(
                            "math"
                        ).sin(angle)
                    )

                    positions[node] = (
                        x,
                        y
                    )

        else:

            # ------------------------------------------------
            # Complete graph layout
            #
            # Arrange each connected component separately
            # instead of forcing the whole graph into one tree.
            # ------------------------------------------------

            components_list = list(
                nx.connected_components(
                    display_graph.to_undirected()
                )
            )

            component_gap = 650

            for component_index, component in enumerate(
                components_list
            ):

                component_nodes = list(
                    component
                )

                component_graph = (
                    display_graph
                    .subgraph(
                        component_nodes
                    )
                    .to_undirected()
                )

                if len(component_nodes) == 1:

                    local_positions = {
                        component_nodes[0]:
                            (0, 0)
                    }

                else:

                    local_positions = (
                        nx.spring_layout(
                            component_graph,
                            seed=42,
                            scale=220
                        )
                    )

                offset_x = (
                    component_index
                    * component_gap
                )

                for node, position in (
                    local_positions.items()
                ):

                    positions[node] = (

                        float(position[0])
                        + offset_x,

                        float(position[1])

                    )

        # ====================================================
        # RELATIONSHIP COLORS
        # ====================================================

        relationship_colors = {

            "IMPACTED":
                "#2563eb",

            "OCCURRED":
                "#7c3aed",

            "OCCURRED_IN":
                "#7c3aed",

            "FOLLOWED_BY":
                "#ea580c",

            "ASSOCIATED_WITH":
                "#dc2626",

            "AT_EXPENSE_OF":
                "#059669",

            "CAUSES":
                "#dc2626",

            "PART_OF":
                "#0891b2",

            "RELATED_TO":
                "#64748b"

        }

        # ====================================================
        # PYVIS SETTINGS
        # ====================================================

        net.set_options(
            """
            {
                "nodes": {

                    "shape": "dot",

                    "borderWidth": 2,

                    "borderWidthSelected": 4,

                    "shadow": {
                        "enabled": true,
                        "color": "rgba(15, 23, 42, 0.18)",
                        "size": 8,
                        "x": 2,
                        "y": 3
                    },

                    "font": {
                        "size": 17,
                        "face": "Arial",
                        "color": "#0f172a",
                        "multi": true
                    }

                },

                "edges": {

                    "width": 2,

                    "selectionWidth": 4,

                    "arrows": {

                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.8
                        }

                    },

                    "font": {

                        "size": 13,

                        "face": "Arial",

                        "color": "#334155",

                        "strokeWidth": 5,

                        "strokeColor": "#ffffff",

                        "align": "middle"

                    },

                    "smooth": {

                        "enabled": true,

                        "type": "curvedCW",

                        "roundness": 0.12

                    }

                },

                "physics": {

                    "enabled": false

                },

                "interaction": {

                    "hover": true,

                    "tooltipDelay": 80,

                    "navigationButtons": true,

                    "keyboard": true,

                    "dragNodes": true,

                    "dragView": true,

                    "zoomView": true,

                    "hideEdgesOnDrag": false,

                    "hideNodesOnDrag": false

                }
            }
            """
        )

        # ====================================================
        # ADD NODES
        # ====================================================

        for node, data in display_graph.nodes(
            data=True
        ):

            node_text = str(
                node
            )

            node_type = str(
                data.get(
                    "type",
                    "Entity"
                )
            )

            level = node_levels.get(
                node,
                0
            )

            # ------------------------------------------------
            # Central Topic
            # ------------------------------------------------

            if (
                center_node is not None
                and node == center_node
            ):

                background = "#dcfce7"

                border = "#15803d"

                node_size = 48

                font_size = 22

                border_width = 4

                node_group = "central"

            # ------------------------------------------------
            # Level 1
            # ------------------------------------------------

            elif level == 1:

                background = "#dbeafe"

                border = "#2563eb"

                node_size = 34

                font_size = 18

                border_width = 3

                node_group = "direct"

            # ------------------------------------------------
            # Level 2+
            # ------------------------------------------------

            else:

                background = "#ffedd5"

                border = "#ea580c"

                node_size = 28

                font_size = 16

                border_width = 2

                node_group = "secondary"

            # ------------------------------------------------
            # Complete graph disconnected components
            # ------------------------------------------------

            if (
                not topic
                or not topic.strip()
            ):

                if (
                    center_node is None
                    or node != center_node
                ):

                    background = "#e0f2fe"

                    border = "#0284c7"

                    node_size = 30

                    font_size = 17

                    border_width = 2

                    node_group = "entity"

            # ------------------------------------------------
            # Wrapped Label
            # ------------------------------------------------

            display_label = wrap_text(
                clean_label(
                    node_text,
                    max_length=34
                ),
                max_length=19
            )

            # ------------------------------------------------
            # Tooltip
            # ------------------------------------------------

            outgoing = []

            incoming = []

            for neighbor in display_graph.successors(
                node
            ):

                relation = str(
                    display_graph.edges[
                        node,
                        neighbor
                    ].get(
                        "relation",
                        "RELATED_TO"
                    )
                ).upper()

                readable_relation = (
                    relation
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )

                outgoing.append(
                    f"{readable_relation} → "
                    f"{neighbor}"
                )

            for predecessor in display_graph.predecessors(
                node
            ):

                relation = str(
                    display_graph.edges[
                        predecessor,
                        node
                    ].get(
                        "relation",
                        "RELATED_TO"
                    )
                ).upper()

                readable_relation = (
                    relation
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )

                incoming.append(
                    f"{predecessor} → "
                    f"{readable_relation}"
                )

            tooltip_lines = [

                f"<b>{node_text}</b>",

                f"Type: {node_type}",

                f"Connections: "
                f"{display_graph.degree(node)}"

            ]

            if (
                topic
                and topic.strip()
            ):

                tooltip_lines.append(
                    f"Graph distance from topic: "
                    f"{level}"
                )

            if outgoing:

                tooltip_lines.append(
                    "<br><b>Outgoing relationships:</b>"
                )

                tooltip_lines.extend(
                    outgoing
                )

            if incoming:

                tooltip_lines.append(
                    "<br><b>Incoming relationships:</b>"
                )

                tooltip_lines.extend(
                    incoming
                )

            tooltip = (
                "<div style='"
                "font-family:Arial;"
                "font-size:13px;"
                "line-height:1.5;"
                "padding:8px;"
                "max-width:400px;"
                "'>"
                + "<br>".join(
                    tooltip_lines
                )
                + "</div>"
            )

            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            x = None

            y = None

            if node in positions:

                x = positions[node][0]

                y = positions[node][1]

            # ------------------------------------------------
            # Add Node
            # ------------------------------------------------

            node_arguments = {

                "label":
                    display_label,

                "title":
                    tooltip,

                "color": {

                    "background":
                        background,

                    "border":
                        border,

                    "highlight": {

                        "background":
                            "#fef3c7",

                        "border":
                            "#ca8a04"

                    },

                    "hover": {

                        "background":
                            "#f8fafc",

                        "border":
                            "#475569"

                    }

                },

                "size":
                    node_size,

                "borderWidth":
                    border_width,

                "font": {

                    "size":
                        font_size,

                    "face":
                        "Arial",

                    "color":
                        "#0f172a",

                    "bold":
                        (
                            node
                            == center_node
                        )

                },

                "group":
                    node_group

            }

            if x is not None and y is not None:

                node_arguments["x"] = x

                node_arguments["y"] = y

                node_arguments["physics"] = False

            net.add_node(

                node,

                **node_arguments

            )

        # ====================================================
        # ADD RELATIONSHIPS
        # ====================================================

        for source, target, data in (
            display_graph.edges(
                data=True
            )
        ):

            relation = str(
                data.get(
                    "relation",
                    "RELATED_TO"
                )
            ).upper()

            readable_relation = (
                relation
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            edge_color = (
                relationship_colors.get(
                    relation,
                    "#64748b"
                )
            )

            net.add_edge(

                source,

                target,

                label=readable_relation,

                title=(
                    f"<b>Relationship:</b> "
                    f"{readable_relation}"
                    f"<br>"
                    f"<b>From:</b> {source}"
                    f"<br>"
                    f"<b>To:</b> {target}"
                ),

                arrows="to",

                width=2.5,

                color={

                    "color":
                        edge_color,

                    "highlight":
                        "#0f172a",

                    "hover":
                        "#334155"

                },

                font={

                    "size":
                        13,

                    "face":
                        "Arial",

                    "color":
                        edge_color,

                    "strokeWidth":
                        5,

                    "strokeColor":
                        "#ffffff",

                    "align":
                        "middle"

                },

                smooth={

                    "enabled":
                        True,

                    "type":
                        "curvedCW",

                    "roundness":
                        0.12

                }

            )

        # ====================================================
        # SAVE HTML
        # ====================================================

        graph_file = os.path.join(

            PROJECT_ROOT,

            "data",

            "kg_visualization.html"

        )

        net.save_graph(
            graph_file
        )

        # ====================================================
        # DISPLAY GRAPH
        # ====================================================

        with open(
            graph_file,
            "r",
            encoding="utf-8"
        ) as file:

            html = file.read()

        components.html(

            html,

            height=720,

            scrolling=True

        )

        # ====================================================
        # RELATIONSHIP DETAILS
        # ====================================================

        st.markdown(
            "### 🔗 Relationships in This Graph"
        )

        relationships = []

        for source, target, data in (
            display_graph.edges(
                data=True
            )
        ):

            relation = str(
                data.get(
                    "relation",
                    "RELATED_TO"
                )
            ).upper()

            readable_relation = (
                relation
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            relationships.append({

                "From":
                    str(source),

                "Relationship":
                    readable_relation,

                "To":
                    str(target)

            })

        if relationships:

            for relationship in relationships:

                col1, col2, col3 = st.columns(
                    [3, 2, 3]
                )

                with col1:

                    st.markdown(
                        f"**{relationship['From']}**"
                    )

                with col2:

                    st.markdown(
                        f"`{relationship['Relationship']}`"
                    )

                with col3:

                    st.markdown(
                        f"**{relationship['To']}**"
                    )

        else:

            st.caption(
                "No relationships are available "
                "for the displayed graph."
            )

        # ====================================================
        # LEGEND
        # ====================================================

        if topic and topic.strip():

            st.markdown(
                "### 🎯 Graph Legend"
            )

            info_col1, info_col2, info_col3 = st.columns(
                3
            )

            with info_col1:

                st.markdown(
                    "**🟢 Topic**"
                )

                st.caption(
                    "The entity selected as the "
                    "current topic."
                )

            with info_col2:

                st.markdown(
                    "**🔵 Direct Entities**"
                )

                st.caption(
                    "Entities directly connected "
                    "to the topic."
                )

            with info_col3:

                st.markdown(
                    "**🟠 Deeper Entities**"
                )

                st.caption(
                    "Entities reached through "
                    "additional relationships."
                )

        st.caption(
            "💡 Hover over a node to inspect its type "
            "and relationships. Hover over an edge to "
            "inspect the relationship between two entities. "
            "Drag nodes and use the mouse wheel to explore "
            "the graph."
        )

    except Exception as e:

        st.error(
            f"Unable to display "
            f"Knowledge Graph: {e}"
        )


# ============================================================
# TOPIC
# ============================================================

st.subheader(
    "📚 Topic"
)

topic = st.text_input(

    "Enter Topic",

    placeholder=(
        "Example: Green Revolution"
    )

)


# ============================================================
# GRAPH DEPTH
# ============================================================

if topic.strip():

    st.subheader(
        "🔍 Knowledge Graph Settings"
    )

    graph_depth = st.slider(

        "Graph Traversal Depth",

        min_value=1,

        max_value=4,

        value=2,

        help=(
            "Controls how many relationship "
            "levels are displayed around "
            "the selected topic."
        )

    )

else:

    graph_depth = 2


# ============================================================
# DISPLAY KNOWLEDGE GRAPH
# ============================================================

display_knowledge_graph(

    topic,

    graph_depth

)


st.divider()


# ============================================================
# QUESTION PAPER PATTERN
# ============================================================

st.subheader(
    "📝 Question Paper Pattern"
)


# ============================================================
# BASIC PAPER DETAILS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    total_marks = st.number_input(

        "Total Marks",

        min_value=1,

        max_value=500,

        value=50,

        step=1

    )


with col2:

    duration = st.text_input(

        "Duration",

        value="2 Hours",

        placeholder=(
            "Example: 2 Hours"
        )

    )


# ============================================================
# NUMBER OF SECTIONS
# ============================================================

num_sections = st.number_input(

    "Number of Sections",

    min_value=1,

    max_value=10,

    value=4,

    step=1

)


st.divider()


# ============================================================
# SECTION CONFIGURATION
# ============================================================

sections = []

calculated_marks = 0


for i in range(
    int(num_sections)
):

    section_name = chr(
        65 + i
    )

    st.markdown(
        f"### Section {section_name}"
    )

    # --------------------------------------------------------
    # Question Type
    # --------------------------------------------------------

    question_type = st.selectbox(

        "Question Type",

        QUESTION_TYPES,

        key=(
            f"question_type_{i}"
        )

    )

    # --------------------------------------------------------
    # Number + Marks
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        question_count = st.number_input(

            "Number of Questions",

            min_value=1,

            max_value=100,

            value=5,

            step=1,

            key=(
                f"question_count_{i}"
            )

        )


    with col2:

        marks_each = st.number_input(

            "Marks per Question",

            min_value=1,

            max_value=50,

            value=1,

            step=1,

            key=(
                f"marks_each_{i}"
            )

        )


    # --------------------------------------------------------
    # Bloom Selection
    # --------------------------------------------------------

    bloom_selection = st.multiselect(

        "Bloom's Taxonomy Levels",

        BLOOM_LEVELS,

        default=[

            "L1 - Remember",

            "L2 - Understand"

        ],

        key=(
            f"blooms_{i}"
        )

    )


    # --------------------------------------------------------
    # Section Marks
    # --------------------------------------------------------

    section_marks = (

        int(question_count)

        * int(marks_each)

    )

    calculated_marks += (
        section_marks
    )


    st.caption(

        f"Section {section_name}: "

        f"{question_count} × "

        f"{marks_each} = "

        f"**{section_marks} marks**"

    )


    # --------------------------------------------------------
    # Save Section
    # --------------------------------------------------------

    sections.append({

        "name":
            section_name,

        "type":
            question_type,

        "count":
            int(question_count),

        "marks_each":
            int(marks_each),

        "blooms":
            bloom_selection

    })


    st.divider()


# ============================================================
# PAPER MARK VALIDATION
# ============================================================

st.subheader(
    "📊 Paper Summary"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(

        "Required Total Marks",

        int(total_marks)

    )


with col2:

    st.metric(

        "Calculated Marks",

        calculated_marks

    )


with col3:

    difference = (

        int(total_marks)

        - calculated_marks

    )

    st.metric(

        "Difference",

        difference

    )


if (
    calculated_marks
    != int(total_marks)
):

    st.warning(

        f"Section marks total "
        f"**{calculated_marks}**, "
        f"but required total is "
        f"**{total_marks}**."

    )

else:

    st.success(
        "Section marks match "
        "the total marks."
    )


# ============================================================
# PAPER PATTERN OBJECT
# ============================================================

paper_pattern = {

    "total_marks":
        int(total_marks),

    "duration":
        duration,

    "sections":
        sections

}


# ============================================================
# VIEW PATTERN
# ============================================================

with st.expander(
    "🔎 View Question Paper Pattern"
):

    st.json(
        paper_pattern
    )


st.divider()


# ============================================================
# GENERATE QUESTION PAPER
# ============================================================

if st.button(

    "📝 Generate Question Paper",

    use_container_width=True

):

    # --------------------------------------------------------
    # Topic Validation
    # --------------------------------------------------------

    if not topic.strip():

        st.warning(
            "Please enter a topic."
        )

    # --------------------------------------------------------
    # Bloom Validation
    # --------------------------------------------------------

    elif any(

        len(
            section["blooms"]
        ) == 0

        for section in sections

    ):

        st.warning(

            "Please select at least "
            "one Bloom's Taxonomy level "
            "for every section."

        )

    # --------------------------------------------------------
    # Marks Validation
    # --------------------------------------------------------

    elif (
        calculated_marks
        != int(total_marks)
    ):

        st.error(

            f"Cannot generate the paper. "

            f"Section marks total "
            f"{calculated_marks}, "

            f"but required total is "
            f"{total_marks}."

        )

    else:

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with st.spinner(

            "Generating Question Paper..."

        ):

            try:

                # --------------------------------------------
                # API Request
                # --------------------------------------------

                response = requests.post(

                    API_URL,

                    json={

                        "topic":
                            topic.strip(),

                        "blooms": [

                            bloom

                            for section
                            in sections

                            for bloom
                            in section["blooms"]

                        ],

                        "paper_pattern":
                            paper_pattern

                    },

                    timeout=600

                )

                # --------------------------------------------
                # Successful Response
                # --------------------------------------------

                if response.status_code == 200:

                    result = (
                        response.json()
                    )

                    paper = result.get(

                        "question_paper",

                        ""

                    )

                    # ----------------------------------------
                    # Empty Response
                    # ----------------------------------------

                    if not paper:

                        st.error(

                            "The API returned "
                            "an empty question paper."

                        )

                    else:

                        # ------------------------------------
                        # Success
                        # ------------------------------------

                        st.success(

                            "Question Paper "
                            "Generated Successfully!"

                        )

                        st.divider()

                        st.subheader(

                            "📝 Generated Question Paper"

                        )

                        # ------------------------------------
                        # Display
                        # ------------------------------------

                        st.text_area(

                            "Question Paper",

                            value=paper,

                            height=900

                        )

                        # ------------------------------------
                        # PDF
                        # ------------------------------------

                        try:

                            pdf_path = (
                                PDFGenerator.generate(
                                    paper
                                )
                            )

                            if (
                                pdf_path
                                and os.path.exists(
                                    pdf_path
                                )
                            ):

                                with open(

                                    pdf_path,

                                    "rb"

                                ) as pdf:

                                    st.download_button(

                                        label=(
                                            "📄 Download PDF"
                                        ),

                                        data=pdf,

                                        file_name=(
                                            "QuestionPaper.pdf"
                                        ),

                                        mime=(
                                            "application/pdf"
                                        ),

                                        use_container_width=True

                                    )

                            else:

                                st.warning(

                                    "PDF was generated "
                                    "but the file could "
                                    "not be found."

                                )

                        except Exception as pdf_error:

                            st.warning(

                                "PDF generation failed: "
                                f"{pdf_error}"

                            )

                # --------------------------------------------
                # API Error
                # --------------------------------------------

                else:

                    st.error(

                        "FastAPI returned status code "
                        f"{response.status_code}"

                    )

                    if response.text:

                        st.code(
                            response.text
                        )

            # ------------------------------------------------
            # Connection Error
            # ------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(

                    "Could not connect to FastAPI. "

                    "Make sure the FastAPI server "
                    "is running."

                )

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(

                    "The request timed out. "

                    "Question generation may "
                    "take some time."

                )

            # ------------------------------------------------
            # Other Error
            # ------------------------------------------------

            except Exception as e:

                st.error(

                    f"Unexpected error: {e}"

                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Hybrid GraphRAG • Knowledge Graph "
    "+ Vector Retrieval + Bloom's Taxonomy "
    "+ Mistral"

)