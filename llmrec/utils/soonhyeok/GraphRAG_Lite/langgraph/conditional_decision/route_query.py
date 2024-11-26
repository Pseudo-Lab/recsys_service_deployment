from ..langgraph_state import Lite_GraphState


def is_general_query(state: Lite_GraphState):
    print(f"Route_query edge".ljust(100, '-'))
    if state['query_type'] == 'general':
        print(f"general")
        return 'NO'
    else:  # 이 외의 대답이 나오면 에러 raise
        print(f"Movie")
        return 'YES'