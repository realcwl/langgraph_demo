from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.modal.generated.joke_pb2 import DadJokeStates


class ModifyDadJokeGraphInput(TypedDict):
    # Take the entire conversation history as input.
    # TODO: Limit this to the last 1 message.
    messages: Annotated[list[AnyMessage], add_messages]

    # The action to take.
    action: Optional[str]

    # Gemini client. Use it in a stateless way.
    worker_agent: ChatVertexAI

    # The state of the dad joke.
    dad_joke_states: DadJokeStates


class ModifyDadJokeGraphOutput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    action: Optional[str]
