from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.graphs.dad_joke.states import DadJokeGraphInput, DadJokeGraphOutput
from app.modal.generated.joke_pb2 import DadJokeStates


async def get_dad_joke_setup(state: DadJokeGraphInput) -> Dict[str, Any]:
    """Get a dad joke setup from the user."""
    tool_call_message = state["messages"][-1]

    assert isinstance(tool_call_message, AIMessage)

    if not tool_call_message.tool_calls:
        raise ValueError("No function call found in the last message")

    arguments = tool_call_message.tool_calls[0]["args"]

    if not arguments.get("topic") or not arguments.get("style"):
        raise ValueError("No topic or style found in the function call arguments")

    return {
        "dad_joke_states": DadJokeStates(
            topic=arguments.get("topic"),
            style=arguments.get("style"),
        ),
    }


async def generate_dad_joke(state: DadJokeGraphInput) -> Dict[str, Any]:
    """Generate a dad joke."""
    assert state["worker_agent"] is not None

    tool_call_message = state["messages"][-1]

    assert isinstance(tool_call_message, AIMessage)
    tool_call_id = tool_call_message.tool_calls[0]["id"]

    dad_joke_states = state["dad_joke_states"]

    # TODO: Refactor this to use a prompt template.
    PROMPT = f"""
    Generate a dad joke with the following topic and style:
    Topic: {dad_joke_states.topic}
    Style: {dad_joke_states.style}
    """

    response = await state["worker_agent"].ainvoke(PROMPT)

    response_message = response.content

    tool_message = ToolMessage(
        content=response_message,
        tool_call_id=tool_call_id,
    )

    return {
        "messages": [tool_message],
    }


def create_dad_joke_graph():
    """Create the dad joke subgraph with HITL."""
    # Define the graph
    builder = StateGraph(DadJokeGraphInput, output=DadJokeGraphOutput)

    # Add nodes
    builder.add_node("generate_setup", get_dad_joke_setup)
    builder.add_node("generate_dad_joke", generate_dad_joke)

    # Add edges
    builder.add_edge(START, "generate_setup")
    builder.add_edge("generate_setup", "generate_dad_joke")
    builder.add_edge("generate_dad_joke", END)  # Return to main graph for HITL

    return builder.compile()
