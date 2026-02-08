from llm_response.langgraph_graph_state import GraphState
from prompt.cypher_tools.attraction import ATTRACTION_CYPHER_PROMPT

def attraction_cypher(llm, state: GraphState) -> dict:
    if state.get("attraction_mentioned") == "":
        return None  # LangGraph 0.2.x: None means no state update

    query = state.get("rewritten_query") or state.get("query")
    prompt = ATTRACTION_CYPHER_PROMPT.format(query=query)
    res = llm.invoke(prompt)
    cypher = res.content.replace("```cypher", "").replace("```", "").strip()

    if cypher:
        return {
            "field_cypher_parts": {"attraction": cypher},
            "field_conditions_summary": {"attraction": "✅ 관광지 조건 추가"}
        }
    return None  # LangGraph 0.2.x: None means no state update
