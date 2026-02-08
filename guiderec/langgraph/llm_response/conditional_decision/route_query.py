from llm_response.langgraph_graph_state import GraphState


def is_search_query(state: GraphState):
    print(f"Route_query edge".ljust(100, '-'))
    if state['query_type'] == 'recomm':
        print(f"recomm")
        return 'NO'
    else:  # 이 외의 대답이 나오면 에러 raise
        print(f"recomm")
        return 'NO'