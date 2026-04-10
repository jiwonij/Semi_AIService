from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict):
    user_query: str

    target_companies: List[str]
    target_technologies: List[str]
    search_queries: List[str]

    raw_documents: List[Dict[str, Any]]
    indexed_documents: List[Dict[str, Any]]

    retrieved_evidence: List[Dict[str, Any]]
    validated_evidence: List[Dict[str, Any]]
    conflicting_evidence: List[Dict[str, Any]]

    query_planning_metrics: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    web_search_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    analysis_metrics: Dict[str, Any]
    trl_metrics: Dict[str, Any]
    report_metrics: Dict[str, Any]

    analysis_result: Optional[Dict[str, Any]]
    trl_result: Optional[Dict[str, Any]]

    report_draft: Optional[str]
    final_report: Optional[str]

    status: str