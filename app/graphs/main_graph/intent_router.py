# Import functions directly from their modules
from app.graphs.main_graph.states import MainState


def route_to_operator(state: MainState) -> str:
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


# Determine if there is function call.
def has_function_call(state: MainState) -> bool:
    """Check if the last message has a function call."""
    action = state["action"]

    if not action:
        return False
    else:
        return True
