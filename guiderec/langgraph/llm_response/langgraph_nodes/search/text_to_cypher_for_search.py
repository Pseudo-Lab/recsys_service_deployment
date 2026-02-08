from llm_response.langgraph_graph_state import GraphState
from prompt.text_to_cypher_for_search import EXAMPLES_COMBINED, NEO4J_SCHEMA, TEXT_TO_CYPHER_FOR_SEARCH_TEMPLATE


def text_to_cypher_for_search(llm, state:GraphState):
    print(f"Text2Cypher for SEARCH".ljust(100, '='))
    response = llm.invoke(
        TEXT_TO_CYPHER_FOR_SEARCH_TEMPLATE.format(
            NEO4J_SCHEMA=NEO4J_SCHEMA,
            EXAMPLES_COMBINED=EXAMPLES_COMBINED, 
            query=state['query']
            )
    )
    # print(f"# input_tokens count : {response.usage_metadata['input_tokens']}")
    cypher = response.content.replace('```', '').replace('cypher', '').strip()
    # print(f"{cypher}")
    
    state['t2c_for_search'] = cypher
    return state
