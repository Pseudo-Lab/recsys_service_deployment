from llm_response.langgraph_graph_state import GraphState
from prompt.cypher_tools.restaurant_name import RESTAURANT_NAME_CYPHER_PROMPT

def restaurant_name_cypher(llm, state: GraphState) -> dict:
    if state.get("restaurant_name_mentioned") == "":
        return None  # LangGraph 0.2.x: None means no state update

    query = state.get("query")
    prompt = RESTAURANT_NAME_CYPHER_PROMPT.format(query=query)
    res = llm.invoke(prompt)
    content = res.content or ""
    cypher = content.replace("```cypher", "").replace("```", "").strip()

    if cypher:
        return {
            "field_cypher_parts": {"restaurant_name": cypher},
            "field_conditions_summary": {"restaurant_name": "✅ 식당명 조건 추가"}
        }
    return None  # LangGraph 0.2.x: None means no state update
