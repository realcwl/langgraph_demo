import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

from app.agent import StatelessAgent
from app.functions.get_dad_joke import get_dad_joke
from app.graphs.main_graph.main_graph import create_main_graph
from app.graphs.main_graph.states import get_agent_response, get_empty_state

os.environ["GOOGLE_CLOUD_PROJECT"] = "hooglee-c2c4d"
os.environ["GOOGLE_CLOUD_REGION"] = "us-central1"

SYSTEM_PROMPT = """You are a helpful assistant that can provide dad jokes and heartwarming poems.
When asked for a dad joke, make it funny but appropriate.
For any other questions, be helpful and concise without calling tools."""


async def main():
    """Main entry point for the assistant."""
    agent = StatelessAgent().set_graph(create_main_graph())
    state = get_empty_state()
    main_loop_client = ChatVertexAI(
        model_name="gemini-2.0-flash", temperature=0.7, max_output_tokens=1024
    )
    main_loop_client_with_tools = main_loop_client.bind_tools([get_dad_joke])
    state["human_interaction_agent"] = main_loop_client_with_tools
    state["worker_agent"] = main_loop_client

    print("Welcome to the LangGraph Demo Assistant!")
    print("You can ask for a dad joke or a heartwarming poem.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke the agent and update the state.
        state = await agent.ainvoke(state)
        print(f"\nAssistant: {get_agent_response(state)}")


if __name__ == "__main__":
    asyncio.run(main())
