from graph.state import GraphState
from services.analysis_service import AnalysisService
from services.query_planning_service import QueryPlanningService
from services.report_service import ReportService
from services.retrieval_service import RetrievalService
from services.trl_service import TRLService
from services.validation_service import ValidationService
from services.web_search_service import WebSearchService


query_planning_service = QueryPlanningService()
retrieval_service = RetrievalService()
web_search_service = WebSearchService()
validation_service = ValidationService()
analysis_service = AnalysisService()
trl_service = TRLService()
report_service = ReportService()


def query_planning_node(state: GraphState):
    result = query_planning_service.run(state["user_query"])
    return {
        "target_companies": result["target_companies"],
        "target_technologies": result["target_technologies"],
        "search_queries": result["search_queries"],
        "query_planning_metrics": result["metrics"],
        "status": "query_planned",
    }


def retrieval_node(state: GraphState):
    result = retrieval_service.run(state["search_queries"])
    return {
        "retrieved_evidence": result["retrieved_evidence"],
        "retrieval_metrics": result["metrics"],
    }


def web_search_node(state: GraphState):
    result = web_search_service.run(state["search_queries"])
    return {
        "raw_documents": result["raw_documents"],
        "web_search_metrics": result["metrics"],
    }


def validation_node(state: GraphState):
    result = validation_service.run(
        retrieved_evidence=state["retrieved_evidence"],
        raw_documents=state["raw_documents"],
    )
    return {
        "validated_evidence": result["validated_evidence"],
        "conflicting_evidence": result["conflicting_evidence"],
        "validation_metrics": result["metrics"],
        "status": "validated",
    }


def analysis_node(state: GraphState):
    result = analysis_service.run(
        validated_evidence=state["validated_evidence"],
        target_companies=state["target_companies"],
        target_technologies=state["target_technologies"],
    )
    return {
        "analysis_result": result["analysis_result"],
        "analysis_metrics": result["metrics"],
        "status": "analyzed",
    }


def trl_node(state: GraphState):
    result = trl_service.run(state["validated_evidence"])
    return {
        "trl_result": result["trl_result"],
        "trl_metrics": result["metrics"],
        "status": "trl_done",
    }


def report_node(state: GraphState):
    result = report_service.run(
        user_query=state["user_query"],
        analysis_result=state["analysis_result"] or {},
        trl_result=state["trl_result"] or {},
        validated_evidence=state["validated_evidence"],
    )
    return {
        "report_draft": result["report_draft"],
        "final_report": result["final_report"],
        "report_metrics": result["metrics"],
        "status": "report_generated",
    }

def supervisor_node(state: GraphState):
    query_pass = state.get("query_planning_metrics", {}).get("pass", False)
    retrieval_pass = state.get("retrieval_metrics", {}).get("pass", False)
    web_pass = state.get("web_search_metrics", {}).get("pass", False)
    validation_pass = state.get("validation_metrics", {}).get("pass", False)
    analysis_pass = state.get("analysis_metrics", {}).get("pass", False)
    trl_pass = state.get("trl_metrics", {}).get("pass", False)
    report_pass = state.get("report_metrics", {}).get("pass", False)

    all_pass = (
        query_pass
        and retrieval_pass
        and web_pass
        and validation_pass
        and analysis_pass
        and trl_pass
        and report_pass
    )

    return {
        "status": "completed" if all_pass else "needs_revision"
    }