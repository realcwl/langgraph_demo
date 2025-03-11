from typing import Any, Dict

from langchain_core.messages import AIMessage

from app.functions.get_dad_joke import get_dad_joke
from app.functions.modify_dad_joke import modify_dad_joke
from app.graphs.main_graph.states import MainState


def _get_dad_joke_args(state: MainState) -> Dict[str, Any]:
    """Get a dad joke setup from the user."""
    tool_call_message = state["messages"][-1]

    assert isinstance(tool_call_message, AIMessage)

    if not tool_call_message.tool_calls:
        raise ValueError("No function call found in the last message")

    arguments = tool_call_message.tool_calls[0]["args"]

    if not arguments.get("topic") or not arguments.get("style"):
        raise ValueError("No topic or style found in the function call arguments")

    topic = arguments.get("topic")
    style = arguments.get("style")

    assert topic is not None and style is not None

    origin_joke = state["dad_joke_states"]
    origin_joke.topic = topic
    origin_joke.style = style

    return {"dad_joke_states": origin_joke}


def _get_modify_dad_joke_args(state: MainState) -> Dict[str, Any]:
    """Get the arguments for the modify dad joke function."""
    tool_call_message = state["messages"][-1]

    assert isinstance(tool_call_message, AIMessage)

    if not tool_call_message.tool_calls:
        raise ValueError("No function call found in the last message")

    arguments = tool_call_message.tool_calls[0]["args"]

    if not arguments.get("instruction"):
        raise ValueError("No instruction found in the function call arguments")

    instruction = arguments.get("instruction")
    assert instruction is not None

    origin_joke = state["dad_joke_states"]
    origin_joke.instruction = instruction

    return {"dad_joke_states": origin_joke}


def extract_args(state: MainState) -> Dict[str, Any]:
    """Extract the arguments from the last message."""
    action = state["action"]

    if action == get_dad_joke.__name__:
        return _get_dad_joke_args(state)

    if action == modify_dad_joke.__name__:
        return _get_modify_dad_joke_args(state)

    return {}
