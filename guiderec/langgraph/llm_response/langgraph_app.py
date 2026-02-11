import os
import sqlite3
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from graphrag.retriever import get_neo4j_vector
from graphrag.graph_search import get_neo4j_vector_graph
from llm_response.conditional_decision.route_query import is_search_query
from llm_response.get_llm_model import get_llm_model
from llm_response.langgraph_graph_state import GraphState
from llm_response.langgraph_nodes.agent.attraction_cypher import attraction_cypher
from llm_response.langgraph_nodes.agent.build_final_cypher_from_parts import build_final_cypher_from_parts
from llm_response.langgraph_nodes.agent.field_detection import field_detection
from llm_response.langgraph_nodes.agent.location_cypher import location_cypher
from llm_response.langgraph_nodes.agent.menu_cypher import menu_cypher
from llm_response.langgraph_nodes.agent.price_cypher import price_cypher
from llm_response.langgraph_nodes.agent.restaurant_name_cypher import restaurant_name_cypher
from llm_response.langgraph_nodes.recommendation.final_formatting_for_recomm import final_formatting_for_recomm
from llm_response.langgraph_nodes.recommendation.similar_menu_store_recomm import similar_menu_store_recomm
from llm_response.langgraph_nodes.search.final_formatting import final_formatting_for_search
from llm_response.langgraph_nodes.recommendation.selecting import final_selecting_for_recomm
from llm_response.langgraph_nodes.recommendation.get_store_candidates import get_store_candidates
from llm_response.langgraph_nodes.search.retrieve_for_search_cypher import retrieve_for_search_cypher
from llm_response.langgraph_nodes.intent_analysis.rewrite import rewrite
from llm_response.langgraph_nodes.intent_analysis.casual_response import casual_response
from llm_response.tools.guiderec_tools import GUIDEREC_TOOLS
from llm_response.tools.tool_executor import GuideRecToolExecutor
from langgraph.graph import END
from guiderec_utils import graphdb_driver
from guiderec_config import CONFIG
from llm_response.langgraph_nodes.search.text_to_cypher_for_search import text_to_cypher_for_search
import json

llm = get_llm_model()
store_retriever_rev_emb = get_neo4j_vector().as_retriever(search_kwargs={"k": CONFIG.store_retriever_rev_emb_k})
store_retriever_grp_emb = get_neo4j_vector_graph().as_retriever(search_kwargs={"k": CONFIG.store_retriever_rev_emb_k_grp})

# Tool Executor 초기화
tool_executor = GuideRecToolExecutor(llm, graphdb_driver, store_retriever_rev_emb, store_retriever_grp_emb)

# LLM with tools bound
llm_with_tools = llm.bind_tools(GUIDEREC_TOOLS)


def tool_agent(state: GraphState) -> dict:
    """LLM이 적절한 tool을 선택합니다."""
    query = state["query"]

    system_prompt = """당신은 제주도 맛집 추천 AI '제주맛집탐험대'입니다.
사용자의 요청에 따라 적절한 도구를 선택하세요:

1. search_restaurant_info: 특정 식당의 정보(주소, 전화번호, 메뉴 등)를 조회할 때
   - 예: "나은이네 주소", "숙성도 전화번호", "OO식당 메뉴"

2. recommend_restaurants: 맛집 추천을 요청할 때
   - 예: "흑돼지 맛집 추천해줘", "성산일출봉 근처 한정식", "2만원대 해산물"

3. casual_chat: 일상 대화, 인사, 감사 등
   - 예: "안녕", "고마워", "뭐해?", "거기 맛있었어", "괜찮은데?"

반드시 하나의 도구를 선택하세요."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = llm_with_tools.invoke(messages)

    # Tool calls 파싱
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

    if tool_calls:
        tool_call = tool_calls[0]  # 첫 번째 tool call 사용
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        print(f"[ToolAgent] Selected tool: {tool_name}, args: {tool_args}")

        return {
            "selected_tool": tool_name,
            "tool_args": tool_args
        }
    else:
        # Tool 선택 안됨 - 기본적으로 casual로 처리
        print(f"[ToolAgent] No tool selected, defaulting to casual")
        return {
            "selected_tool": "casual_chat",
            "tool_args": {"message": query}
        }


def execute_tool(state: GraphState) -> dict:
    """선택된 tool을 실행합니다."""
    tool_name = state.get("selected_tool", "")
    tool_args = state.get("tool_args", {})

    if tool_name == "search_restaurant_info":
        result = tool_executor.execute({
            "name": tool_name,
            "args": tool_args
        })
        return {"final_answer": result}

    elif tool_name == "casual_chat":
        result = tool_executor.execute({
            "name": tool_name,
            "args": tool_args
        })
        return {"final_answer": result}

    elif tool_name == "recommend_restaurants":
        # 추천은 기존 파이프라인으로 - 이 노드에서는 처리하지 않음
        return {}

    return {"final_answer": "무슨 말인지 잘 모르겠어~ 다시 말해줄래? 🍊"}


def route_after_tool_agent(state: GraphState) -> str:
    """Tool 선택 결과에 따라 다음 노드 결정"""
    tool_name = state.get("selected_tool", "")

    if tool_name == "recommend_restaurants":
        return "rewrite"  # 기존 추천 파이프라인으로
    else:
        return "execute_tool"  # search 또는 casual은 바로 실행


workflow = StateGraph(GraphState)

# Nodes
## Tool-based Entry Point
workflow.add_node("tool_agent", tool_agent)
workflow.add_node("execute_tool", execute_tool)

## Rewrite
workflow.add_node("rewrite", lambda state: rewrite(llm, state))

# Field Detection
workflow.add_node("field_detection", lambda state: field_detection(llm, state))
workflow.add_node("restaurant_name_cypher", lambda state: restaurant_name_cypher(llm, state))
workflow.add_node("price_cypher", lambda state: price_cypher(llm, state))
workflow.add_node("location_cypher", lambda state: location_cypher(llm, state))
workflow.add_node("menu_cypher", lambda state: menu_cypher(llm, state))
workflow.add_node("attraction_cypher", lambda state: attraction_cypher(llm, state))
workflow.add_node("build_final_cypher_from_parts", lambda state: build_final_cypher_from_parts(llm, state))


## Search query nodes
# workflow.add_node("text_to_cypher_for_search", lambda state: text_to_cypher_for_search(llm, state))
# workflow.add_node("retrieve_for_search_cypher", lambda state: retrieve_for_search_cypher(graphdb_driver, state))
# workflow.add_node("final_formatting_for_search", lambda state: final_formatting_for_search(llm, graphdb_driver, state))

## Recomm query nodes
workflow.add_node("get_store_candidates", lambda state: get_store_candidates(llm, graphdb_driver, store_retriever_rev_emb, store_retriever_grp_emb, state))
workflow.add_node("final_selecting_for_recomm", lambda state: final_selecting_for_recomm(llm, state))
workflow.add_node("similar_menu_store_recomm", lambda state: similar_menu_store_recomm(graphdb_driver, state))
workflow.add_node("final_formatting_for_recomm", lambda state: final_formatting_for_recomm(graphdb_driver, state))

# Edges
# Tool Agent - Conditional routing based on selected tool
workflow.add_conditional_edges(
    "tool_agent",
    route_after_tool_agent,
    {
        "execute_tool": "execute_tool",
        "rewrite": "rewrite"
    }
)

# execute_tool (search/casual) goes to END
workflow.add_edge("execute_tool", END)

# Agent
workflow.add_edge('rewrite', 'field_detection')

# Fan-out: field_detection 이후 5개 cypher 노드 병렬 실행
workflow.add_edge("field_detection", "restaurant_name_cypher")
workflow.add_edge("field_detection", "price_cypher")
workflow.add_edge("field_detection", "location_cypher")
workflow.add_edge("field_detection", "menu_cypher")
workflow.add_edge("field_detection", "attraction_cypher")

# Fan-in: 모든 cypher 노드 완료 후 build_final_cypher_from_parts 실행
workflow.add_edge("restaurant_name_cypher", "build_final_cypher_from_parts")
workflow.add_edge("price_cypher", "build_final_cypher_from_parts")
workflow.add_edge("location_cypher", "build_final_cypher_from_parts")
workflow.add_edge("menu_cypher", "build_final_cypher_from_parts")
workflow.add_edge("attraction_cypher", "build_final_cypher_from_parts")

workflow.add_edge("build_final_cypher_from_parts", "get_store_candidates")

## Search
# workflow.add_edge('text_to_cypher_for_search', 'retrieve_for_search_cypher')
# workflow.add_edge('retrieve_for_search_cypher', 'final_formatting_for_search')
# workflow.add_edge('final_formatting_for_search', END)

## Recomm
workflow.add_edge('get_store_candidates', 'final_selecting_for_recomm')
workflow.add_edge('final_selecting_for_recomm', 'similar_menu_store_recomm')
workflow.add_edge('similar_menu_store_recomm', 'final_formatting_for_recomm')
workflow.add_edge('final_formatting_for_recomm', END)

workflow.set_entry_point("tool_agent")

# SQLite Checkpointer 설정
# EC2에서는 /home/ec2-user/recsys_service_deployment/guiderec_checkpoints.db 사용
CHECKPOINTS_DB_PATH = os.environ.get(
    "GUIDEREC_CHECKPOINTS_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "guiderec_checkpoints.db")
)

try:
    # SQLite connection with check_same_thread=False for multi-threaded access
    _sqlite_conn = sqlite3.connect(CHECKPOINTS_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(_sqlite_conn)
    app = workflow.compile(checkpointer=checkpointer)
    print(f"[GuideRec] Compiled with SQLite checkpointer: {CHECKPOINTS_DB_PATH}")
except Exception as e:
    print(f"[GuideRec] Failed to initialize checkpointer: {e}, running without checkpointer")
    app = workflow.compile()
