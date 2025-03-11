from typing import Optional


def get_motivational_quote(source: Optional[str] = None, theme: Optional[str] = None) -> str:
    """
    Get a motivational quote, optionally from a specific source or about a specific theme.

    Args:
        source: Optional source of the quote (e.g., 'Chinese history', 'famous athletes', 'philosophers')
        theme: Optional theme for the quote (e.g., 'perseverance', 'success', 'happiness')

    Returns:
        A motivational quote matching the requested parameters
    """
    # This function will be called by the LLM but actual implementation
    # happens in the workflow
    return "The only way to do great work is to love what you do."
