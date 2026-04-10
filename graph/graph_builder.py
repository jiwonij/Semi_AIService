from langgraph.graph import END, StateGraph

from graph.nodes import (
    analysis_node,
    query_planning_node,
    report_node,
    retrieval_node,
    supervisor_node,
    trl_node,
    validation_node,
    web_search_node,
)
from graph.state import GraphState


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("query_planning", query_planning_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("validation", validation_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("trl", trl_node)
    builder.add_node("report", report_node)
    builder.add_node("supervisor", supervisor_node)

    builder.set_entry_point("query_planning")

    builder.add_edge("query_planning", "retrieval")
    builder.add_edge("query_planning", "web_search")
    builder.add_edge("retrieval", "validation")
    builder.add_edge("web_search", "validation")
    builder.add_edge("validation", "analysis")
    builder.add_edge("analysis", "trl")
    builder.add_edge("trl", "report")
    builder.add_edge("report", "supervisor")
    builder.add_edge("supervisor", END)

    return builder.compile()