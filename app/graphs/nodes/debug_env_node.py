from typing import Any, Dict

from langchain_google_vertexai import ChatVertexAI

from app.functions.get_dad_joke import get_dad_joke
from app.functions.modify_dad_joke import modify_dad_joke
from app.graphs.main_graph.states import MainState


def initialize_debug_env(state: MainState) -> Dict[str, Any]:
    """Extract the arguments from the last message."""
    main_loop_client = ChatVertexAI(
        model_name="gemini-2.0-flash", temperature=0.7, max_output_tokens=1024
    )
    main_loop_client_with_tools = main_loop_client.bind_tools([get_dad_joke, modify_dad_joke])

    return {
        "human_interaction_agent": main_loop_client_with_tools,
        "worker_agent": main_loop_client,
    }
