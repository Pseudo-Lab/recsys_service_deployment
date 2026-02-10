import json
import sys
import os

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain_core.runnables import RunnableConfig

# Add langgraph path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langgraph'))

# Lazy loading for Neo4j-dependent components
_guiderec_app = None
_GraphState = None
GUIDEREC_AVAILABLE = False

# Node name to Korean description mapping
NODE_DESCRIPTIONS = {
    'intent_router': ('의도 파악', '질문의 의도를 파악하고 있어요'),
    'casual_response': ('응답 생성', '응답을 생성하고 있어요'),
    'rewrite': ('쿼리 분석', '질문을 분석하고 있어요'),
    'field_detection': ('조건 파악', '원하시는 조건을 파악하고 있어요'),
    'restaurant_name_cypher': ('식당명 검색', '식당명을 확인하고 있어요'),
    'price_cypher': ('가격대 분석', '가격 조건을 분석하고 있어요'),
    'location_cypher': ('위치 분석', '위치 조건을 분석하고 있어요'),
    'menu_cypher': ('메뉴 분석', '메뉴 조건을 분석하고 있어요'),
    'attraction_cypher': ('관광지 연계', '주변 관광지를 확인하고 있어요'),
    'build_final_cypher_from_parts': ('검색 쿼리 생성', '맛집 검색 쿼리를 만들고 있어요'),
    'get_store_candidates': ('맛집 후보 검색', '조건에 맞는 맛집을 찾고 있어요'),
    'final_selecting_for_recomm': ('맛집 선정', 'AI가 최적의 맛집을 선별하고 있어요'),
    'similar_menu_store_recomm': ('유사 맛집 검색', '비슷한 메뉴의 다른 맛집도 찾고 있어요'),
    'final_formatting_for_recomm': ('결과 정리', '추천 결과를 정리하고 있어요'),
}


def get_dynamic_description(node_name, state):
    """노드별 동적 설명 생성"""
    if not state:
        return None

    if node_name == 'get_store_candidates':
        count = state.get('candidates_count', 0)
        if count > 0:
            return f'후보 맛집 {count}개 찾았어요!'
        else:
            # 개수가 없으면 candidate_str에서 추정
            candidate_str = state.get('candidate_str', '')
            if candidate_str:
                count = candidate_str.count('pk:')
                if count > 0:
                    return f'후보 맛집 {count}개 찾았어요!'

    elif node_name == 'final_selecting_for_recomm':
        recommendations = state.get('selected_recommendations', {})
        if recommendations and 'recommendations' in recommendations:
            count = len(recommendations['recommendations'])
            return f'최적의 맛집 {count}개를 선정했어요!'

    elif node_name == 'similar_menu_store_recomm':
        sim_pks = state.get('sim_recomm_pks', [])
        if sim_pks:
            return f'유사한 맛집 {len(sim_pks)}개도 찾았어요!'

    return None


def _init_guiderec():
    """Lazy initialization of GuideRec LangGraph app"""
    global _guiderec_app, _GraphState, GUIDEREC_AVAILABLE
    if _guiderec_app is None:
        try:
            print("[GuideRec] Initializing LangGraph app...")
            from guiderec.langgraph.llm_response.langgraph_app import app
            from guiderec.langgraph.llm_response.langgraph_graph_state import GraphState
            _guiderec_app = app
            _GraphState = GraphState
            GUIDEREC_AVAILABLE = True
            print("[GuideRec] Initialization successful!")
        except Exception as e:
            import traceback
            print(f"[GuideRec] Initialization failed: {e}")
            traceback.print_exc()
            GUIDEREC_AVAILABLE = False
    return _guiderec_app, _GraphState


def guiderec_home(request):
    """GuideRec 홈페이지"""
    context = {
        'title': 'Jeju Food Guide',
        'description': '제주도 맛집을 AI가 추천해드립니다. 여행 동행과 연령대에 맞는 맞춤형 추천!',
    }
    return render(request, 'guiderec/home.html', context)


@csrf_exempt
def guiderec_chat(request):
    """GuideRec 채팅 페이지"""
    if request.method == 'GET':
        context = {
            'title': 'Jeju Food Guide',
            'description': '제주도 맛집 추천 AI와 대화해보세요!',
            'initial_message': '''안녕하세요! 제주도 맛집을 찾고 계신가요? 🍊<br>
여행 동행(가족, 친구, 연인 등)과 원하시는 음식 종류를 알려주세요!<br><br>
<span style="font-size:0.85rem; color:rgba(255,255,255,0.85);">예시) 클릭해서 바로 질문해보세요!</span><br>
<span class="example-query" onclick="useExample(this)">부모님과 성산일출봉 근처에서 3만원대 한정식 먹고 싶어요</span><br>
<span class="example-query" onclick="useExample(this)">친구들이랑 한라산 등산 후 갈만한 흑돼지 맛집 추천해줘</span>''',
        }
        return render(request, 'guiderec/chat.html', context)

    # POST: 채팅 메시지 처리 (스트리밍)
    app, GraphState = _init_guiderec()

    if app is None:
        return JsonResponse({
            'status': 'error',
            'message': '현재 이 기능은 사용할 수 없습니다. Neo4j 서버 연결이 필요합니다.'
        })

    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('message', {})
        query = message.get('text', '')

        print(f"[GuideRec] Query: {query}")

        def event_stream():
            """Generator function that yields SSE events"""
            try:
                config = RunnableConfig(recursion_limit=20, configurable={"thread_id": "guiderec"})
                graph_state = GraphState(query=query, messages=[])

                current_step = 0
                total_steps = len(NODE_DESCRIPTIONS)
                last_node = None
                final_answer = None
                is_casual = False  # 일반 대화인지 여부

                # LangGraph 스트리밍 실행
                for chunk in app.stream(graph_state, config=config):
                    # chunk는 {node_name: state} 형태
                    for node_name, state in chunk.items():
                        # casual_response면 상태바 없이 바로 결과 반환
                        if node_name == 'casual_response':
                            is_casual = True
                            if state and 'final_answer' in state and state['final_answer']:
                                final_answer = state['final_answer']
                            continue

                        # intent_router는 상태바에 표시하지 않음
                        if node_name == 'intent_router':
                            continue

                        if node_name != last_node:
                            # 첫 진행 상태일 때 시작 이벤트 전송
                            if last_node is None:
                                yield f"data: {json.dumps({'type': 'start', 'message': '질문을 이해하고 있어요...'}, ensure_ascii=False)}\n\n"

                            last_node = node_name
                            current_step += 1
                            progress = int((current_step / total_steps) * 100)

                            step_name, step_desc = NODE_DESCRIPTIONS.get(
                                node_name,
                                (node_name, f'{node_name} 처리 중...')
                            )

                            # 동적 설명이 있으면 사용
                            dynamic_desc = get_dynamic_description(node_name, state)
                            if dynamic_desc:
                                step_desc = dynamic_desc

                            # 진행 상황 이벤트
                            yield f"data: {json.dumps({'type': 'progress', 'step': step_name, 'description': step_desc, 'progress': progress, 'node': node_name}, ensure_ascii=False)}\n\n"

                            print(f"[GuideRec] Node: {node_name} ({progress}%) - {step_desc}")

                            # field_detection 완료 후 감지된 조건 전송
                            if node_name == 'field_detection' and state:
                                conditions = state.get('field_conditions_summary', {})
                                if conditions:
                                    yield f"data: {json.dumps({'type': 'conditions', 'conditions': conditions}, ensure_ascii=False)}\n\n"
                                    print(f"[GuideRec] Detected conditions: {conditions}")

                        # 최종 결과 저장
                        if state and 'final_answer' in state and state['final_answer']:
                            final_answer = state['final_answer']

                # 완료 이벤트 (일반 대화가 아닐 때만)
                if not is_casual:
                    yield f"data: {json.dumps({'type': 'progress', 'step': '완료', 'description': '추천이 완료되었습니다!', 'progress': 100}, ensure_ascii=False)}\n\n"

                if final_answer:
                    yield f"data: {json.dumps({'type': 'result', 'status': 'success', 'message': final_answer, 'is_casual': is_casual}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': '추천 결과를 생성하지 못했습니다.', 'is_casual': is_casual}, ensure_ascii=False)}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response

    except json.JSONDecodeError as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
    except Exception as e:
        print(f"[GuideRec] Error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})
