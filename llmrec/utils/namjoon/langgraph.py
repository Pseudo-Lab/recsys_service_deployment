from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional, Sequence
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)
from utils import queryAnalysis, queryAnalRouter
from websearch import WebSearchNode
from expertised import ExpertisedNode
from chat import ChatMessagesNode
from ticket import TicketNode
from qa import QANode, qaRouter


class RecsysState(TypedDict):
    question: str                         # Query - 유저 input
    # question_type: str                    # Query의 형식 (복합쿼리, 단순 쿼리 등 -> 디테일을 위해선 routing된 후가 나을 수 있음)
    # sub_question: List[str]               # Query anal을 통해 세분화된 유저 input
    user_id: Optional[str]                # 유저 식별자
    user_history: Optional[List[str]]               # 유저 이용내역
    web_search_result: Optional[str]      # web search 결과값
    expertised_search: Optional[str]      # 평론가 평 등
    chat_messages: str                    # 일반 채팅    
    recommendations: Optional[List[str]]  # 추천 결과
    num_rec: Optional[int]                # 추천 개수
    reservation_status: Optional[str]     # 예매 정보
    messages: Annotated[Sequence[BaseMessage], operator.add]      # 진행 내역 저장


def GraphGenerate():
    graph = StateGraph(RecsysState)
    graph.add_node("query", queryAnalysis)
    graph.add_node("web", WebSearchNode)
    graph.add_node("experts", ExpertisedNode)
    graph.add_node("chat", ChatMessagesNode)
    graph.add_node("ticket", TicketNode)
    graph.add_node("qa", QANode)
    graph.add_node("recommendation", RecommendationNode)
    
    
    # Set the Starting Edge
    graph.set_entry_point("query")

    # Set our Contitional Edges
    graph.add_conditional_edges(
        "query",
        queryAnalRouter,
        {
            "chat": "chat",
            "recommendation": "recommendation",
            "reservation": "ticket",
            "expertised": "experts",
            "web": "web",
        },
    )
    graph.add_edge("chat", END)
    graph.add_edge("recommendation", "qa")
    graph.add_edge("ticket", "qa")
    graph.add_edge("experts", "qa")
    graph.add_conditional_edges(
        "qa",
        qaRouter,
        {
            "web": "web",
            "expertised": "experts",
            "recommendation": "recommendation",
            "reservation": "ticket",
            "SUCCESS": END
        },
    )
    # Compile the workflow
    app = graph.compile()
    return app 

def run(question):
    app = GraphGenerate()
    response = app.invoke({
    "messages": [
        HumanMessage(
            content=question
        )
    ]})
    return response