from ..langgraph_state import GraphState


def is_general_query(state: GraphState):
    print(f"Route_query edge".ljust(100, '-'))
    if state['query_type'] == 'general':
        print(f"general")
        return 'NO'
    else:  # 이 외의 대답이 나오면 에러 raise
        print(f"NoGeneral")
        return 'YES'