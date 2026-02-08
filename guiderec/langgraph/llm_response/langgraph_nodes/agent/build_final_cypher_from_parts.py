import re
from llm_response.langgraph_graph_state import GraphState
from prompt.cypher_tools.final_cypher_gen import FINAL_CYPHER_GENERATION_PROMPT


def fix_deprecated_cypher_syntax(cypher: str) -> str:
    """Neo4j 5.x에서 deprecated된 문법을 수정"""
    # exists(variable.property) -> variable.property IS NOT NULL
    pattern = r'exists\s*\(\s*(\w+)\.(\w+)\s*\)'
    replacement = r'\1.\2 IS NOT NULL'
    return re.sub(pattern, replacement, cypher, flags=re.IGNORECASE)


def build_final_cypher_from_parts(llm, state: GraphState) -> GraphState:
    parts = state.get("field_cypher_parts", {})
    query = state["query"]

    print("\n" + "=" * 100)
    print("FINAL CYPHER GENERATION".center(100))
    print("=" * 100)
    print(f"Query             : {query}")
    print(f"Detected Fields   : {', '.join(parts.keys()) if parts else 'None'}")
    print("-" * 100)

    print("Individual Field Conditions:")
    if parts:
        for k, v in parts.items():
            print(f"[{k.upper()}]".ljust(15) + "=> " + v.strip().replace("\n", " "))
    else:
        print("❌ No field-level Cypher conditions detected.")
    print("=" * 100)

    prompt = FINAL_CYPHER_GENERATION_PROMPT.format(
        query=query,
        parts="\n\n".join([f'{k}' + '\n' + f'{v}' for k, v in parts.items()])
    )
    print(f"prompt : {prompt}")
    res = llm.invoke(prompt)
    final_cypher = res.content.strip().replace("```", "").replace("cypher", "")

    # Neo4j 5.x deprecated 문법 수정
    final_cypher = fix_deprecated_cypher_syntax(final_cypher)

    print("FINAL CYPHER QUERY".center(100))
    print("-" * 100)
    print(final_cypher)
    print("=" * 100 + "\n")

    state["t2c_for_recomm"] = final_cypher
    return state
