import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from .predictor import CredibilityPredictor
from .retriever import FactCheckRetriever
from .reasoner import CredibilityReasoner
from .report_generator import generate_pdf

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    article_text: str
    prediction: dict
    retrieval: dict
    analysis: dict
    pdf_report: Optional[bytes]
    error: str



_predictor: Optional[CredibilityPredictor] = None
_retriever: Optional[FactCheckRetriever] = None
_reasoner: Optional[CredibilityReasoner] = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = CredibilityPredictor()
    return _predictor


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = FactCheckRetriever()
    return _retriever


def _get_reasoner():
    global _reasoner
    if _reasoner is None:
        _reasoner = CredibilityReasoner()
    return _reasoner




def predict_node(state: AgentState) -> dict:
    try:
        result = _get_predictor().predict(state["article_text"])
        return {"prediction": result}
    except Exception as e:
        logger.error(f"predict_node failed: {e}")
        return {"prediction": {}, "error": str(e)}


def retrieve_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    try:
        result = _get_retriever().retrieve(state["article_text"])
        return {"retrieval": result}
    except Exception as e:
        logger.warning(f"retrieve_node failed (continuing): {e}")
        return {"retrieval": {"keywords_used": "", "sources": [], "method": "none"}}


def reason_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    try:
        result = _get_reasoner().analyze(
            state["article_text"],
            state["prediction"],
            state["retrieval"],
        )
        return {"analysis": result}
    except Exception as e:
        logger.error(f"reason_node failed: {e}")
        return {"analysis": {}, "error": str(e)}


def report_node(state: AgentState) -> dict:
    if state.get("error") or not state.get("analysis"):
        return {"pdf_report": None}
    try:
        pdf_bytes = generate_pdf(
            state["article_text"],
            state["prediction"],
            state["analysis"],
            state["retrieval"],
        )
        return {"pdf_report": pdf_bytes}
    except Exception as e:
        logger.warning(f"report_node failed (PDF skipped): {e}")
        return {"pdf_report": None}




def build_agent():
    g = StateGraph(AgentState)

    g.add_node("predict", predict_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason", reason_node)
    g.add_node("report", report_node)

    g.set_entry_point("predict")
    g.add_edge("predict", "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", "report")
    g.add_edge("report", END)

    return g.compile()

credibility_agent = build_agent()


def run_agent(article_text: str) -> AgentState:
    """
    Entry point for the UI team.
    Pass the raw article text; get back a fully populated AgentState dict.
    Keys: article_text, prediction, retrieval, analysis, pdf_report, error
    """
    if not article_text or not article_text.strip():
        raise ValueError("article_text cannot be empty")

    initial: AgentState = {
        "article_text": article_text,
        "prediction": {},
        "retrieval": {},
        "analysis": {},
        "pdf_report": None,
        "error": "",
    }
    return credibility_agent.invoke(initial)
