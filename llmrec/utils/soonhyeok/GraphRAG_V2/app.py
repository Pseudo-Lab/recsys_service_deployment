from .langgraph.langgraph_app import soonhyeok_app, Soonhyeok_GraphState
from langchain_core.runnables import RunnableConfig


def get_results(query, session_id, state:Soonhyeok_GraphState) : 
    """
    사용자 입력(query)과 대화 히스토리(messages)를 바탕으로 LangGraph를 호출하여 결과를 반환.

    Args:
        query (str): 사용자가 입력한 질문.
        messages (list): 대화 히스토리. 각 항목은 {"role": "user"/"assistant", "content": str} 형태.
        thread_id (str): LangGraph 스레드 ID (default: "default_thread").

    Returns:
        dict: LangGraph 호출 결과.
            - final_answer (str): 최종 답변.
    """
    try:
        # LangGraph 설정
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": session_id})

        # GraphState 생성
        graph_state = Soonhyeok_GraphState(query=query)

        # LangGraph 호출
        result_gs = soonhyeok_app.invoke(graph_state, config=config)

        # 결과 처리
        final_answer = result_gs.get('final_answer', "죄송합니다. 답변을 생성하지 못했습니다.")

        return state

    except Exception as e:
        print(f"Exception 발생: {str(e)}")
        return {
            "final_answer": "오류가 발생했습니다. 다시 시도해주세요.",
            "error": str(e)
        }