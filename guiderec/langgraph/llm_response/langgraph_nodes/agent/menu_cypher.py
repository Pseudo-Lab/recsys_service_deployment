from llm_response.langgraph_graph_state import GraphState
from prompt.cypher_tools.menu import MENU_CYPHER_PROMPT

def menu_cypher(llm, state: GraphState) -> dict:
    if state.get("menu_mentioned") == "":
        return None  # LangGraph 0.2.x: None means no state update

    query = state.get("rewritten_query") or state.get("query")
    prompt = MENU_CYPHER_PROMPT.format(query=query)
    res = llm.invoke(prompt)
    content = res.content or ""
    cypher = content.replace("```cypher", "").replace("```", "").strip()

    if cypher:
        return {
            "field_cypher_parts": {"menu": cypher},
            "field_conditions_summary": {"menu": "✅ 메뉴 조건 추가"}
        }
    return None  # LangGraph 0.2.x: None means no state update
