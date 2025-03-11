from typing import Optional


def get_dad_joke(topic: Optional[str] = None, style: Optional[str] = None) -> str:
    """
    Get a dad joke, optionally about a specific topic or in a specific style. Always provide arguments.

    Args:
        topic: Optional topic for the dad joke must be choosing from ['food', 'animals', 'work'].
        style: Optional style for the joke must be choosing from ['clean', 'silly', 'pun'], default is 'clean'.

    Returns:
        A dad joke matching the requested parameters.
    """
    # This function will be called by the LLM but actual implementation
    # happens in the workflow
    return ""
