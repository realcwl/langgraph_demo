from langgraph.graph.state import CompiledStateGraph

from app.graphs.main_graph.states import MainState


class StatelessAgent:
    def __init__(self):
        pass

    def set_graph(self, graph: CompiledStateGraph) -> "StatelessAgent":
        self.graph = graph
        return self

    # Asynchronously invoke the graph on a single input.
    async def ainvoke(self, state: MainState) -> MainState:
        result = await self.graph.ainvoke(state)
        return MainState(**result)
