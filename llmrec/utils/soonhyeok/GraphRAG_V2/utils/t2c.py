from ..langgraph.langgraph_state import Soonhyeok_GraphState
from ..prompt.t2c_recomm import EXAMPLES_COMBINED, NEO4J_SCHEMA_RECOMM, TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE


def text_to_cypher_for_recomm(llm, state:Soonhyeok_GraphState):
    print(f"Text2Cypher for RECOMM".ljust(100, '-'))
    response = llm.invoke(
        TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE.format(
            NEO4J_SCHEMA_RECOMM=NEO4J_SCHEMA_RECOMM,
            EXAMPLES_COMBINED=EXAMPLES_COMBINED, 
            query=state['query']
            )
    )
    cypher = response.content.replace('```', '').replace('cypher', '').strip()
    print(f"# cypher : \n{cypher}\n")
    state['t2c_for_recomm'] = cypher
    print(f"# input_tokens count : {response.usage_metadata['input_tokens']}")
    print(f"".ljust(100, '-'))
    return state