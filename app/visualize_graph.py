from app.graphs.main_graph.main_graph import create_main_graph

create_main_graph().get_graph(xray=True).draw_mermaid_png(output_file_path="main_graph.png")
