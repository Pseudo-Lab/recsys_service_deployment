import json
import re
import time
from llm_response.langgraph_graph_state import GraphState
from prompt.final_selecting_for_recomm import FINAL_SELECTING_FOR_RECOMM_v2
from pprint import pprint

MAX_RETRIES = 2


def parse_llm_response(raw: str) -> dict:
    """LLM 응답을 파싱하여 dict로 반환"""
    # 코드 블록/주석 제거
    cleaned = raw.replace("```", "").replace("json", "").strip()

    # 작은따옴표를 큰따옴표로 변환
    cleaned_json = re.sub(r"'", '"', cleaned)

    # 여러 파싱 방법 시도
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        pass

    # json.loads 실패 시 eval 시도 (fallback)
    try:
        return eval(cleaned)
    except:
        pass

    raise ValueError(f"JSON 파싱 실패: {cleaned[:200]}...")


def final_selecting_for_recomm(llm, state: GraphState):
    print("Selecting for recomm".ljust(100, '='))
    print(f"state['query'] : {state['query']}")
    print()
    print(f"state['rewritten_query'] : {state['rewritten_query']}")
    print()
    print(f"state['candidate_str'] : {state['candidate_str'][:500]}...")
    print()

    prompt = FINAL_SELECTING_FOR_RECOMM_v2.format(
        query=state['query'],
        intent=state['rewritten_query'],
        candidates=state['candidate_str']
    )

    # 토큰 수 추정 (대략 4자당 1토큰)
    prompt_chars = len(prompt)
    estimated_input_tokens = prompt_chars // 4
    print(f"[Timing] Prompt chars: {prompt_chars}, Estimated input tokens: ~{estimated_input_tokens}")

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            llm_start = time.time()
            response = llm.invoke(prompt)
            llm_elapsed = time.time() - llm_start

            output_chars = len(response.content)
            estimated_output_tokens = output_chars // 4
            print(f"[Timing] LLM call took: {llm_elapsed:.2f}s (attempt {attempt + 1})")
            print(f"[Timing] Output chars: {output_chars}, Estimated output tokens: ~{estimated_output_tokens}")

            raw = response.content.strip()
            state["selected_recommendations"] = parse_llm_response(raw)
            return state

        except Exception as e:
            last_error = e
            print(f"[Retry] 파싱 실패 (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
            if attempt < MAX_RETRIES:
                print(f"[Retry] 재시도 중...")
                time.sleep(0.5)

    # 모든 시도 실패
    raise ValueError(f"LLM 응답 파싱 최종 실패: {last_error}")
