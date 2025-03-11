from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.graphs.modify_dad_joke.states import ModifyDadJokeGraphInput, ModifyDadJokeGraphOutput


async def modify_dad_joke(state: ModifyDadJokeGraphInput) -> Dict[str, Any]:
    """Modify a dad joke."""
    assert state["worker_agent"] is not None

    tool_call_message = state["messages"][-1]

    assert isinstance(tool_call_message, AIMessage)
    tool_call_id = tool_call_message.tool_calls[0]["id"]

    dad_joke_states = state["dad_joke_states"]

    # TODO: Refactor this to use a prompt template.
    PROMPT = f"""
    Modify the dad joke based on user's instructions:
    Original joke: {dad_joke_states.content}
    Instructions: {dad_joke_states.instruction}
    """

    response = await state["worker_agent"].ainvoke(PROMPT)

    response_message = response.content

    tool_message = ToolMessage(
        content=response_message,
        tool_call_id=tool_call_id,
    )

    dad_joke_states.content = str(response_message)

    return {
        "messages": [tool_message],
        "dad_joke_states": dad_joke_states,
    }


def create_modify_dad_joke_graph():
    """Create the dad joke subgraph with HITL."""
    # Define the graph
    builder = StateGraph(ModifyDadJokeGraphInput, output=ModifyDadJokeGraphOutput)

    # Add nodes
    builder.add_node("modify_dad_joke", modify_dad_joke)

    # Add edges
    builder.add_edge(START, "modify_dad_joke")
    builder.add_edge("modify_dad_joke", END)  # Return to main graph for HITL

    return builder.compile()
