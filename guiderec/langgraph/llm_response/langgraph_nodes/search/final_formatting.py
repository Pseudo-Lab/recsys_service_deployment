from llm_response.langgraph_graph_state import GraphState
from prompt.final_formatting_for_search import FINAL_FORMATTING_FOR_SEARCH
from datetime import datetime


def final_formatting_for_search(llm, graphdb_driver, state: GraphState):
    print(f"Final formatting for search".ljust(100, '='))
    response = llm.invoke(
      FINAL_FORMATTING_FOR_SEARCH.format(
         query=state['query'],
         cypher=state['t2c_for_search'],
         search_result=str(state['record_dict_lst'])
         )
    )
    print(response.content)
    print(state.keys())

    state["final_answer"] = response.content
    return state
