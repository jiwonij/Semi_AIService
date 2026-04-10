from graph.graph_builder import build_graph


def main():
    graph = build_graph()

    initial_state = {
        "user_query": "HBM technology trend and competitor strategy",
        "target_companies": [],
        "target_technologies": [],
        "search_queries": [],
        "raw_documents": [],
        "indexed_documents": [],
        "retrieved_evidence": [],
        "validated_evidence": [],
        "conflicting_evidence": [],
        "query_planning_metrics": {},
        "retrieval_metrics": {},
        "web_search_metrics": {},
        "validation_metrics": {},
        "analysis_metrics": {},
        "trl_metrics": {},
        "report_metrics": {},
        "analysis_result": None,
        "trl_result": None,
        "report_draft": None,
        "final_report": None,
        "status": "start",
    }

    result = graph.invoke(initial_state)

    print("최종 상태:", result["status"])
    print("타겟 기업:", result.get("target_companies"))
    print("타겟 기술:", result.get("target_technologies"))
    print("검색 쿼리 수:", len(result.get("search_queries", [])))
    print("PDF 저장 경로:", result.get("final_report"))
    print("query planning metrics:", result.get("query_planning_metrics"))

    print("retrieval metrics:", result.get("retrieval_metrics"))
    print("web search metrics:", result.get("web_search_metrics"))
    print("validation metrics:", result.get("validation_metrics"))
    print("analysis metrics:", result.get("analysis_metrics"))
    print("trl metrics:", result.get("trl_metrics"))
    print("report metrics:", result.get("report_metrics"))


if __name__ == "__main__":
    main()