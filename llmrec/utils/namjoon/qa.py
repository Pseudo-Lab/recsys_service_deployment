def QANode(state):
    """
    쿼리에 맞게 결과값이 생성되었는지 제확인합니다.
    """

def qaRouter(state):
    """
    오리지날 쿼리에 맞게 되었으면 계속 진행,
    아니라면 다시 작업처리 합니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if 'SUCCESS' in last_message:
        return 'SUCCESS'
    if 'web' in last_message:
        return 'web'
    if 'expertised' in last_message:
        return 'expertised'
    if 'recommendation' in last_message:
        return 'recommendation'
    if 'reservation' in last_message:
        return 'reservation'
