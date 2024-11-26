from ..langgraph.langgraph_state import Soonhyeok_GraphState
from ..prompt.general_prompt import GENERAL_PROMPT_TEMPLATE

def general_for_recomm(llm, state:Soonhyeok_GraphState):
    print(f"general for RECOMM".ljust(100, '-'))
    response = llm.invoke(
        GENERAL_PROMPT_TEMPLATE.format(
            query=state['query']
            )
    )
    print("response: ", response)
    state["final_answer"] = response.content
    print(f"# final_answer : \n{state['final_answer']}\n")
    print(f"# input_tokens count : {response.usage_metadata['input_tokens']}")

    
    return state