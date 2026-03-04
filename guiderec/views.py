import json
import sys
import os
import uuid
import time

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from .models import ChatSession, ChatMessage
from .event_logger import event_logger

# Add langgraph path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langgraph'))

# Lazy loading for Neo4j-dependent components
_guiderec_app = None
_GraphState = None
GUIDEREC_AVAILABLE = False

# Node name to Korean description mapping
NODE_DESCRIPTIONS = {
    'tool_agent': ('도구 선택', '어떤 도움이 필요한지 파악하고 있어요'),
    'execute_tool': ('정보 조회', '요청하신 정보를 찾고 있어요'),
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
    'final_formatting_for_recomm': ('결과 정리', '추천 결과를 정리하고 있어요'),
}


def get_dynamic_description(node_name, state):
    """노드별 동적 설명 생성"""
    if not state:
        return None

    if node_name == 'get_store_candidates':
        # candidate_str에서 pk: 개수로 후보 수 추정 (스트리밍에서 더 신뢰할 수 있음)
        candidate_str = state.get('candidate_str', '') if hasattr(state, 'get') else ''
        count = candidate_str.count('pk:') if candidate_str else 0

        # fallback으로 candidates_count 확인
        if not count:
            count = state.get('candidates_count', 0) if hasattr(state, 'get') else 0

        if count and count > 0:
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
        # Get session_id from query params
        session_id = request.GET.get('session_id')
        previous_messages = []

        if request.user.is_authenticated and session_id:
            try:
                session = ChatSession.objects.get(
                    id=session_id,
                    user=request.user,
                    is_active=True
                )
                previous_messages = list(session.messages.values('role', 'content'))
            except ChatSession.DoesNotExist:
                session_id = None

        context = {
            'title': 'Jeju Food Guide',
            'description': '제주도 맛집 추천 AI와 대화해보세요!',
            'initial_message': '''안녕하세요! 제주도 맛집을 찾고 계신가요? 🍊<br>
여행 동행(가족, 친구, 연인 등)과 원하시는 음식 종류를 알려주세요!<br><br>
<span style="font-size:0.85rem; color:rgba(255,255,255,0.85);">예시) 클릭해서 바로 질문해보세요!</span><br>
<span class="example-query" onclick="useExample(this)">부모님과 성산일출봉 근처에서 3만원대 한정식 먹고 싶어요</span><br>
<span class="example-query" onclick="useExample(this)">친구들이랑 한라산 등산 후 갈만한 흑돼지 맛집 추천해줘</span>''',
            'session_id': session_id,
            'previous_messages': json.dumps(previous_messages, ensure_ascii=False),
            'is_authenticated': request.user.is_authenticated,
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
        session_id = data.get('session_id')

        print(f"[GuideRec] Query: {query}, Session: {session_id}")

        # Get or create session (로그인/비로그인 모두 지원)
        session = None
        if session_id:
            try:
                if request.user.is_authenticated:
                    session = ChatSession.objects.get(id=session_id, user=request.user)
                else:
                    # 익명 사용자도 세션 ID로 조회 (user=null인 세션)
                    session = ChatSession.objects.get(id=session_id, user__isnull=True)
            except ChatSession.DoesNotExist:
                session = None

        if not session:
            # 새 세션 생성 (로그인 시 user 연결, 비로그인 시 user=null)
            if request.user.is_authenticated:
                session = ChatSession.objects.create(user=request.user)
            else:
                session = ChatSession.objects.create(user=None)
            session_id = str(session.id)

        # Save user message (로그인/비로그인 모두)
        ChatMessage.objects.create(
            session=session,
            role='user',
            content=query
        )

        # Generate title from first message
        if session.title == 'New Chat':
            session.generate_title_from_first_message()

        # Use session ID as thread_id for LangGraph checkpointer
        thread_id = session_id if session_id else str(uuid.uuid4())

        # 이전 대화 메시지 로드 (최근 10개) - 현재 메시지 제외
        previous_messages = []
        prev_msgs = session.messages.order_by('-created_at')[1:11]  # 현재 메시지 제외
        for msg in reversed(prev_msgs):
            previous_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        def event_stream():
            """Generator function that yields SSE events"""
            nonlocal session  # Access session from outer scope
            try:
                langfuse_handler = LangfuseCallbackHandler(
                    session_id=thread_id,
                    user_id=str(request.user.id) if request.user.is_authenticated else None,
                    metadata={"query": query}
                )
                config = RunnableConfig(recursion_limit=20, configurable={"thread_id": thread_id}, callbacks=[langfuse_handler])
                graph_state = GraphState(query=query, messages=previous_messages)

                current_step = 0
                total_steps = len(NODE_DESCRIPTIONS)
                last_node = None
                final_answer = None
                is_casual = False  # 일반 대화인지 여부
                node_start_time = time.time()  # 노드별 시간 측정
                request_start_time = time.time()  # 전체 요청 시간 측정
                node_timings = {}  # 노드별 소요시간 기록
                candidates_count = 0  # 후보 개수
                detected_conditions = {}  # 감지된 조건
                rewritten = None  # 확장된 쿼리

                # S3 이벤트 로깅: 요청 시작
                event_logger.log(
                    "guiderec_request_start",
                    session_id=thread_id,
                    query=query,
                    user_id=str(request.user.id) if request.user.is_authenticated else None,
                    is_authenticated=request.user.is_authenticated,
                    message_count=len(previous_messages) + 1
                )

                # LangGraph 스트리밍 실행
                for chunk in app.stream(graph_state, config=config):
                    # chunk는 {node_name: state} 형태
                    for node_name, state in chunk.items():
                        # execute_tool (search/casual)은 상태바 없이 바로 결과 반환
                        if node_name == 'execute_tool':
                            is_casual = True  # 상태바 없이 처리
                            if state and 'final_answer' in state and state['final_answer']:
                                final_answer = state['final_answer']
                            continue

                        # casual_response면 상태바 없이 바로 결과 반환 (기존 호환)
                        if node_name == 'casual_response':
                            is_casual = True
                            if state and 'final_answer' in state and state['final_answer']:
                                final_answer = state['final_answer']
                            continue

                        # tool_agent, intent_router, similar_menu_store_recomm는 상태바에 표시하지 않음
                        if node_name in ('tool_agent', 'intent_router', 'similar_menu_store_recomm'):
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

                            # 노드별 실행 시간 측정
                            node_elapsed = time.time() - node_start_time
                            node_timings[node_name] = round(node_elapsed, 2)
                            print(f"[GuideRec] Node: {node_name} ({progress}%) - {step_desc} [{node_elapsed:.1f}s]")
                            node_start_time = time.time()  # 다음 노드 측정 시작

                            # rewrite 완료 후 확장된 쿼리 전송
                            if node_name == 'rewrite' and state:
                                rewritten = state.get('rewritten_query', '')
                                if rewritten and rewritten != query:
                                    yield f"data: {json.dumps({'type': 'rewritten_query', 'original': query, 'rewritten': rewritten}, ensure_ascii=False)}\n\n"
                                    print(f"[GuideRec] Rewritten query: {rewritten}")
                                    # S3 이벤트 로깅: 쿼리 확장
                                    event_logger.log(
                                        "guiderec_query_rewritten",
                                        session_id=thread_id,
                                        original_query=query,
                                        rewritten_query=rewritten
                                    )

                            # field_detection 완료 후 감지된 조건 전송
                            if node_name == 'field_detection' and state:
                                detected_conditions = state.get('field_conditions_summary', {})
                                if detected_conditions:
                                    yield f"data: {json.dumps({'type': 'conditions', 'conditions': detected_conditions}, ensure_ascii=False)}\n\n"
                                    print(f"[GuideRec] Detected conditions: {detected_conditions}")
                                    # S3 이벤트 로깅: 조건 추출
                                    event_logger.log(
                                        "guiderec_conditions_detected",
                                        session_id=thread_id,
                                        conditions=detected_conditions,
                                        condition_count=len(detected_conditions)
                                    )

                            # get_store_candidates 완료 후 후보 개수 업데이트
                            if node_name == 'get_store_candidates' and state:
                                # 디버그: state의 실제 키 출력
                                state_keys = list(state.keys()) if hasattr(state, 'keys') else []
                                print(f"[DEBUG] get_store_candidates state keys: {state_keys}")

                                # candidate_str에서 pk: 개수로 후보 수 추정
                                candidate_str = state.get('candidate_str', '') if hasattr(state, 'get') else ''
                                print(f"[DEBUG] candidate_str length: {len(candidate_str) if candidate_str else 0}")
                                if candidate_str:
                                    print(f"[DEBUG] candidate_str sample: {candidate_str[:500]}...")

                                count = candidate_str.count('pk:') if candidate_str else 0

                                # candidates_count도 확인
                                if not count:
                                    count = state.get('candidates_count', 0) if hasattr(state, 'get') else 0
                                    print(f"[DEBUG] candidates_count: {count}")

                                if count and count > 0:
                                    candidates_count = count
                                    yield f"data: {json.dumps({'type': 'update_step', 'description': f'후보 맛집 {count}개 찾았어요!'}, ensure_ascii=False)}\n\n"
                                    print(f"[GuideRec] Candidates count: {count}")
                                    # S3 이벤트 로깅: 후보 검색
                                    event_logger.log(
                                        "guiderec_candidates_found",
                                        session_id=thread_id,
                                        candidates_count=count
                                    )
                                else:
                                    print(f"[DEBUG] No candidates count found, count={count}")


                        # 최종 결과 저장
                        if state and 'final_answer' in state and state['final_answer']:
                            final_answer = state['final_answer']

                # 완료 이벤트 (일반 대화가 아닐 때만)
                if not is_casual:
                    yield f"data: {json.dumps({'type': 'progress', 'step': '완료', 'description': '추천이 완료되었습니다!', 'progress': 100}, ensure_ascii=False)}\n\n"

                # Save assistant message (로그인/비로그인 모두)
                if final_answer and session:
                    ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=final_answer,
                        metadata={'is_casual': is_casual}
                    )
                    session.save()  # Update timestamp

                # 전체 요청 시간 로그
                total_elapsed = time.time() - request_start_time
                print(f"[GuideRec] Total request time: {total_elapsed:.1f}s")

                # S3 이벤트 로깅: 요청 완료
                event_logger.log(
                    "guiderec_request_complete",
                    session_id=thread_id,
                    status="success" if final_answer else "error",
                    is_casual=is_casual,
                    candidates_count=candidates_count,
                    total_time_seconds=round(total_elapsed, 2),
                    node_timings=node_timings,
                    conditions=detected_conditions,
                    rewritten_query=rewritten
                )

                if final_answer:
                    yield f"data: {json.dumps({'type': 'result', 'status': 'success', 'message': final_answer, 'is_casual': is_casual, 'session_id': thread_id}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': '추천 결과를 생성하지 못했습니다.', 'is_casual': is_casual, 'session_id': thread_id}, ensure_ascii=False)}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                # S3 이벤트 로깅: 에러
                event_logger.log(
                    "guiderec_request_error",
                    session_id=thread_id,
                    error_message=str(e),
                    error_type=type(e).__name__
                )
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


# ============================================================
# Session Management API Endpoints
# ============================================================

@csrf_exempt
@require_http_methods(["GET"])
def session_list(request):
    """Get list of chat sessions for the current user."""
    if not request.user.is_authenticated:
        return JsonResponse({'sessions': [], 'authenticated': False})

    sessions = ChatSession.objects.filter(
        user=request.user,
        is_active=True
    ).values('id', 'title', 'updated_at')[:50]

    # Convert UUID to string for JSON serialization
    sessions_list = [
        {
            'id': str(s['id']),
            'title': s['title'],
            'updated_at': s['updated_at'].isoformat() if s['updated_at'] else None
        }
        for s in sessions
    ]

    return JsonResponse({
        'sessions': sessions_list,
        'authenticated': True
    })


@csrf_exempt
@require_http_methods(["POST"])
def session_create(request):
    """Create a new chat session."""
    if not request.user.is_authenticated:
        # Anonymous session - just return a temporary ID
        session_id = str(uuid.uuid4())
        return JsonResponse({
            'session_id': session_id,
            'title': 'New Chat',
            'authenticated': False
        })

    session = ChatSession.objects.create(
        user=request.user,
        title='New Chat'
    )

    return JsonResponse({
        'session_id': str(session.id),
        'title': session.title,
        'authenticated': True
    })


@csrf_exempt
@require_http_methods(["GET"])
def session_detail(request, session_id):
    """Get session details with messages."""
    try:
        if request.user.is_authenticated:
            session = ChatSession.objects.get(
                id=session_id,
                user=request.user,
                is_active=True
            )
        else:
            return JsonResponse({'error': 'Not authenticated'}, status=401)

        messages = list(session.messages.values('role', 'content', 'metadata', 'created_at'))
        for msg in messages:
            msg['created_at'] = msg['created_at'].isoformat() if msg['created_at'] else None

        return JsonResponse({
            'session_id': str(session.id),
            'title': session.title,
            'messages': messages
        })
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)


@csrf_exempt
@require_http_methods(["POST", "DELETE"])
def session_delete(request, session_id):
    """Soft delete a chat session."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        session.is_active = False
        session.save(update_fields=['is_active'])
        return JsonResponse({'success': True})
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)
