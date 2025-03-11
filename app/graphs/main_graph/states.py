from typing import Annotated

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.modal.generated.joke_pb2 import DadJokeStates


# Define the state for the main graph.
# You should always avoid using Optional, unless it's absolutely necessary.
# Having something only ocassionally set is error prone.
class MainState(TypedDict):
    # A list of messages tracking the conversation.
    messages: Annotated[list[AnyMessage], add_messages]

    # Action to take.
    action: str

    # Gemini client. Use it in a stateless way.
    human_interaction_agent: Runnable[LanguageModelInput, BaseMessage]

    # Worker should call this agent.
    worker_agent: ChatVertexAI

    # Data to be used for the dad joke.
    dad_joke_states: DadJokeStates


def get_empty_state(
    human_interaction_agent: Runnable[LanguageModelInput, BaseMessage],
    worker_agent: ChatVertexAI,
) -> MainState:
    return MainState(
        messages=[],
        action="",
        human_interaction_agent=human_interaction_agent,
        worker_agent=worker_agent,
        dad_joke_states=DadJokeStates(),
    )


def get_agent_response(state: MainState) -> str:
    """Get the system response from the state."""
    response = state["messages"][-1]
    if not isinstance(response, AIMessage):
        raise ValueError(f"Last message is not an AIMessage, got {response}")

    return response.content  # type: ignore
