"""
GuideRec Tool Agent 대화 테스트
- 10개 시나리오, 각 5턴 연속 대화
- LLM이 후속 대화 생성
- 메모리 및 응답시간 측정
"""

import requests
import json
import time
import psutil
import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv('.env.dev')

BASE_URL = "http://localhost:8000/guiderec/chat/"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_memory_usage():
    """현재 프로세스 메모리 사용량 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def chat(message: str, session_id: str = None) -> dict:
    """GuideRec API 호출"""
    start_time = time.time()

    try:
        response = requests.post(
            BASE_URL,
            json={"message": {"text": message}, "session_id": session_id},
            stream=True,
            timeout=120
        )

        final_answer = None
        selected_tool = None
        new_session_id = session_id

        # SSE 파싱
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if data.get('type') == 'result':
                            final_answer = data.get('message', '')
                            new_session_id = data.get('session_id', session_id)
                        elif data.get('type') == 'progress':
                            node = data.get('node', '')
                            if node == 'tool_agent':
                                pass  # tool 선택은 로그에서 확인
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - start_time

        return {
            'success': True,
            'answer': final_answer or "응답 없음",
            'session_id': new_session_id,
            'time': elapsed,
            'memory': get_memory_usage()
        }

    except Exception as e:
        return {
            'success': False,
            'answer': f"에러: {str(e)}",
            'session_id': session_id,
            'time': time.time() - start_time,
            'memory': get_memory_usage()
        }


def generate_follow_up(conversation_history: list, bot_response: str) -> str:
    """LLM으로 다음 사용자 발화 생성"""
    history_str = "\n".join([
        f"{'User' if i%2==0 else 'Bot'}: {msg}"
        for i, msg in enumerate(conversation_history)
    ])

    prompt = f"""당신은 제주도 맛집을 찾는 사용자입니다.
아래 대화 맥락을 보고, 자연스러운 후속 질문이나 대화를 1문장으로 생성하세요.

대화 기록:
{history_str}
Bot: {bot_response}

규칙:
- 맥락에 맞는 자연스러운 후속 질문
- 때로는 추가 정보 요청 (주소, 메뉴, 가격 등)
- 때로는 감사 표현이나 일상 대화
- 1문장만 출력

User:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.8
    )

    return response.choices[0].message.content.strip()


def run_conversation_test(test_num: int, start_message: str, turns: int = 5):
    """단일 테스트 실행 (5턴 대화)"""
    print(f"\n{'='*60}")
    print(f"테스트 #{test_num}: \"{start_message}\"")
    print('='*60)

    conversation = []
    session_id = None
    results = []

    current_message = start_message

    for turn in range(turns):
        print(f"\n[턴 {turn+1}] User: {current_message}")

        # API 호출
        result = chat(current_message, session_id)
        session_id = result['session_id']

        print(f"       Time: {result['time']:.2f}s | Memory: {result['memory']:.1f}MB")
        print(f"       Bot: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")

        conversation.append(current_message)
        conversation.append(result['answer'])

        results.append({
            'turn': turn + 1,
            'user': current_message,
            'bot': result['answer'],
            'time': result['time'],
            'memory': result['memory'],
            'success': result['success']
        })

        # 마지막 턴이 아니면 후속 대화 생성
        if turn < turns - 1:
            current_message = generate_follow_up(conversation, result['answer'])
            time.sleep(0.5)  # API 과부하 방지

    return results


def run_all_tests():
    """10개 테스트 시나리오 실행"""
    test_scenarios = [
        "안녕",
        "나은이네 주소 알려줘",
        "흑돼지 맛집 추천해줘",
        "고마워 잘먹을게",
        "숙성도 메뉴 뭐있어?",
        "부모님과 성산일출봉 근처 한정식",
        "ㅎㅇㅎㅇ",
        "해산물 먹고싶어",
        "제주시 2만원대 고기집",
        "중문 근처 분위기 좋은 카페",
    ]

    all_results = []

    print("\n" + "="*60)
    print("GuideRec Tool Agent 대화 테스트 시작")
    print(f"총 {len(test_scenarios)}개 시나리오, 각 5턴 대화")
    print("="*60)

    for i, scenario in enumerate(test_scenarios, 1):
        results = run_conversation_test(i, scenario, turns=5)
        all_results.append({
            'test_num': i,
            'start_message': scenario,
            'results': results
        })

        # 테스트 간 휴식
        if i < len(test_scenarios):
            print("\n--- 다음 테스트 준비 중 (3초) ---")
            time.sleep(3)

    # 최종 결과 요약
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)

    total_turns = 0
    total_time = 0
    success_count = 0

    for test in all_results:
        test_time = sum(r['time'] for r in test['results'])
        test_success = all(r['success'] for r in test['results'])
        total_turns += len(test['results'])
        total_time += test_time
        if test_success:
            success_count += 1

        status = "✅" if test_success else "❌"
        print(f"{status} 테스트 #{test['test_num']}: \"{test['start_message'][:20]}...\" | {test_time:.1f}s")

    print(f"\n총 {total_turns}턴 | 총 시간: {total_time:.1f}s | 성공: {success_count}/{len(test_scenarios)}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
