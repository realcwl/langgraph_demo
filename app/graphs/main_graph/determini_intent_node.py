from typing import Any, Dict

from langchain_core.messages import AIMessage

from app.graphs.main_graph.states import MainState


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
