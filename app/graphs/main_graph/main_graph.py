from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

# Import functions directly from their modules
from app.graphs.dad_joke.dad_joke_graph import create_dad_joke_graph
from app.graphs.main_graph.states import MainState


def custom_router(state: MainState) -> str:
    """
    Determines which node to route to based on user input.
    - If it's a math expression, go to `calculator`.
    - If it's a general question, go to `search`.
    - If the user says 'exit' or 'stop', end the conversation.
    - Otherwise, let the LLM handle the response.
    """
    action = state["action"]

    if not action:
        return "end"
    else:
        return action


async def determine_intent(state: MainState) -> Dict[str, Any]:
    """Determine the user's intent and extract parameters using function calling.

    With tools now registered via LLM.bind_tool, we can rely on the client to
    automatically include the function schemas for get_dad_joke.
    """
    assert state["human_interaction_agent"] is not None

    # Calling the LLM asynchronously.
    response = await state["human_interaction_agent"].ainvoke(state["messages"])

    action = None

    assert isinstance(response, AIMessage)
    if response.tool_calls:
        action = response.tool_calls[0]["name"]

    if response.additional_kwargs.get("function_call"):
        action = response.additional_kwargs["function_call"].get("name")
    return {
        "messages": [response],
        "action": action,
    }


def create_main_graph():
    """Create the main graph that coordinates the workflows."""
    # Define the graph
    builder = StateGraph(MainState)

    # Add nodes
    builder.add_node("determine_intent", determine_intent)
    builder.add_node("get_dad_joke_node", create_dad_joke_graph())

    # Add edges
    builder.add_edge(START, "determine_intent")
    # Add a loopback edge.
    builder.add_edge("get_dad_joke_node", "determine_intent")

    # Fix the conditional edges
    builder.add_conditional_edges(
        "determine_intent",
        custom_router,
        {
            "get_dad_joke": "get_dad_joke_node",
            "end": END,
        },
    )

    return builder.compile()
