from llm_response.langgraph_graph_state import GraphState
from prompt.cypher_tools.location import LOCATION_CYPHER_PROMPT

def location_cypher(llm, state: GraphState) -> dict:
    if state.get("location_mentioned") == "":
        return None  # LangGraph 0.2.x: None means no state update

    query = state.get("rewritten_query") or state.get("query")
    prompt = LOCATION_CYPHER_PROMPT.format(query=query)
    res = llm.invoke(prompt)
    cypher = res.content.replace("```cypher", "").replace("```", "").strip()

    if cypher:
        return {
            "field_cypher_parts": {"location": cypher},
            "field_conditions_summary": {"location": "✅ 위치 조건 추가"}
        }
    return None  # LangGraph 0.2.x: None means no state update
