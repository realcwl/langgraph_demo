from typing import Annotated, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Define the state for the main graph.
class MainState(TypedDict):
    # A list of messages tracking the conversation.
    messages: Annotated[list[AnyMessage], add_messages]

    # Action to take.
    action: Optional[str]

    # Gemini client. Use it in a stateless way.
    human_interaction_agent: Optional[Runnable[LanguageModelInput, BaseMessage]]

    # Worker should call this agent.
    worker_agent: Optional[ChatVertexAI]


def get_empty_state() -> MainState:
    return MainState(messages=[], action=None, human_interaction_agent=None, worker_agent=None)


def get_agent_response(state: MainState) -> str:
    """Get the system response from the state."""
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise ValueError(f"Last message is not an AIMessage, got {response}")

    return response.content  # type: ignore
