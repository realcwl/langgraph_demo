from langgraph.graph import END, START, StateGraph

# Import functions directly from their modules
from app.functions.get_dad_joke import get_dad_joke
from app.functions.modify_dad_joke import modify_dad_joke
from app.graphs.create_dad_joke.dad_joke_graph import create_dad_joke_graph
from app.graphs.main_graph.determini_intent_node import determine_intent
from app.graphs.main_graph.extract_args_node import extract_args
from app.graphs.main_graph.intent_router import has_function_call, route_to_operator
from app.graphs.main_graph.states import MainState
from app.graphs.modify_dad_joke.modify_dad_joke_graph import create_modify_dad_joke_graph


def create_main_graph():
    """Create the main graph that coordinates the workflows."""
    # Define the graph
    builder = StateGraph(MainState)

    # Add nodes
    builder.add_node("determine_intent", determine_intent)
    builder.add_node("extract_args", extract_args)
    builder.add_node("get_dad_joke_node", create_dad_joke_graph())
    builder.add_node("modify_dad_joke_node", create_modify_dad_joke_graph())
    # Add edges
    builder.add_edge(START, "determine_intent")
    builder.add_conditional_edges(
        "determine_intent",
        has_function_call,
        {
            True: "extract_args",
            False: END,
        },
    )
    builder.add_conditional_edges(
        "extract_args",
        route_to_operator,
        {
            get_dad_joke.__name__: "get_dad_joke_node",
            modify_dad_joke.__name__: "modify_dad_joke_node",
        },
    )
    # Add a loopback edge once when the get_dad_joke_node is finished.
    builder.add_edge("get_dad_joke_node", "determine_intent")
    builder.add_edge("modify_dad_joke_node", "determine_intent")

    return builder.compile()
