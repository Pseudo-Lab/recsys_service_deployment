import json

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from clients import MysqlClient
from db_clients.dynamodb import DynamoDBClient
from llmrec.utils.gyungah.load_chain import get_chain as g_get_chain
from llmrec.utils.hyeonwoo.load_chain import router
from llmrec.utils.kyeongchan.get_model import kyeongchan_model
from llmrec.utils.kyeongchan.langgraph_test import GraphState, app
# Conditional import for Neo4j-dependent GraphRAG_V2
try:
    from llmrec.utils.soonhyeok.GraphRAG_V2.langgraph.langgraph_app import soonhyeok_app, Soonhyeok_GraphState
    NEO4J_GRAPHRAG_V2_AVAILABLE = True
except Exception as e:
    print(f"GraphRAG_V2 (Neo4j) not available: {e}")
    soonhyeok_app = None
    Soonhyeok_GraphState = None
    NEO4J_GRAPHRAG_V2_AVAILABLE = False
from llmrec.utils.soonhyeok.GraphRAG_Lite.langgraph.langgraph_app import lite_app, Lite_GraphState
from llmrec.utils.kyeongchan.search_engine import SearchManager
from llmrec.utils.kyeongchan.utils import get_landing_page_recommendation
from llmrec.utils.log_questions import log_llm

from movie.utils import (get_interacted_movie_dicts, get_user_logs_df,
                         get_username_sid, log_tracking)

mysql = MysqlClient()
load_dotenv(".env.dev")
table_llm = DynamoDBClient(table_name="llm")
table_clicklog = DynamoDBClient(table_name="clicklog")


@csrf_exempt
def pplrec(request):
    log_tracking(request=request, view="pplrec")
    username, session_id = get_username_sid(request, _from="llmrec/pplrec GET")
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            message = data.get("message", "")

            # 여기서 message를 원하는 대로 처리
            question = message.get("text")
            log_llm(request=request, question=question, model_name="pplrec")
            print(
                f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}"
            )
            new_response = "준비중이예요오오오"
            log_llm(request=request, answer=new_response, model_name="pplrec")

            return JsonResponse({"status": "success", "message": new_response})
        except json.JSONDecodeError as e:
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        user_logs_df = get_user_logs_df(username, session_id)
        interacted_movie_d = get_interacted_movie_dicts(user_logs_df)
        context = {
            "description1": "Pseudorec's Personalized LLM Recommendation",
            "description2": "슈도렉 멤버가 다같이 만드는 영화 A-Z LLM 모델",
            "watched_movie": interacted_movie_d,
        }
        return render(request, "llmrec_pplrec.html", context)


@csrf_exempt
def llmrec_hyeonwoo(request):
    log_tracking(request=request, view="hyeonwoo")
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            message = data.get("message", "")

            # 여기서 message를 원하는 대로 처리
            question = message.get("text")
            log_llm(request=request, question=question, model_name="hyeonwoo")
            print(
                f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}"
            )
            new_response = router(question)
            log_llm(request=request, answer=new_response, model_name="hyeonwoo")

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse(
                {
                    "status": "success",
                    "message": new_response,
                    "url": "/llmrec/hyeonwoo/",
                }
            )
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        context = {
            "description1": "Hyeonwoo's LLMREC",
            "description2": "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 코난이에요!🎬✨ <br>명탐정으로서 여러분들의 요구사항을 해결할게요 🕵️",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_namjoon(request):
    log_tracking(request=request, view="namjoon")
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            message = data.get("message", "")
            question = message.get("text")
            log_llm(request=request, question=question, model_name="namjoon")

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(
                f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}"
            )

            response_message = "[남준님 모델]아직 모델이 없어요ㅠ"
            log_llm(request=request, answer=response_message, model_name="namjoon")

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse(
                {
                    "status": "success",
                    "message": response_message,
                    "url": "/llmrec/namjoon/",
                }
            )
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        context = {
            "description1": "Namjoon's LLMREC",
            "description2": "남준님의 모델소개 : 준비중입니다!",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_kyeongchan(request):
    username, session_id = get_username_sid(request, _from="llmrec_kyeongchan")
    if request.method == "GET":
        log_tracking(request=request, view="kyeongchan")
        user_logs_df = get_user_logs_df(username, session_id)
        print(f"user_logs_df : \n{user_logs_df}")
        if len(user_logs_df):
            interacted_movie_d = get_interacted_movie_dicts(user_logs_df)
            context = {
                "description1": "깃잔심팀 - LLM 영화 개인화 추천",
                # 'description2': "Self-Querying RAG 기법을 사용한 추천",
                # 'initial_message': f'{username}의 취향을 분석 중입니다...',
                "initial_message": "홈화면에서 좋아하는 영화 평점을 매긴 후 질문을 해보세요! 최근에 별점을 매겼거나 클릭한 영화를 기반으로 개인화 추천을 합니다.<br>감독 또는 배우의 이름, 시놉시스를 언급해주시면 답변을 잘합니다!<br><br>예시) 봉준호 감독이 연출한 영화 추천해줘<br>예시) 레오나르도 디카프리오가 출연한 영화 추천해줘",
                "watched_movie": interacted_movie_d,
            }
        else:
            context = {
                "description1": "깃잔심팀 - LLM 영화 개인화 추천",
                # 'description2': "Self-Querying RAG 기법을 사용한 추천",
                # 'initial_message': f'{username}의 취향을 분석 중입니다...',
                "initial_message": "홈화면에서 좋아하는 영화 평점을 매긴 후 질문을 해보세요! 최근에 별점을 매겼거나 클릭한 영화를 기반으로 개인화 추천을 합니다.<br>감독 또는 배우의 이름, 시놉시스를 언급해주시면 답변을 잘합니다!<br><br>예시) 봉준호 감독이 연출한 영화 추천해줘<br>예시) 레오나르도 디카프리오가 출연한 영화 추천해줘",
                "watched_movie": [],
            }

        return render(request, "llmrec_kyeongchan.html", context)
    else:
        data = json.loads(request.body.decode("utf-8"))
        message = data.get("message", "")["text"]
        print(f"질문 : {message}")
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "movie"})
        inputs = GraphState(question=message, username=username)
        response_message = app.invoke(inputs, config=config)
        return JsonResponse(
            {"status": "success", "message": response_message["final_answer"]}
        )


@csrf_exempt
def llmrec_minsang(request):
    log_tracking(request=request, view="minsang")
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            message = data.get("message", "")
            question = message.get("text")
            log_llm(request=request, question=question, model_name="minsang")

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(
                f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}"
            )

            response_message = "[민상님 모델]아직 모델이 없어요ㅠ"
            log_llm(request=request, answer=response_message, model_name="minsang")

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({"status": "success", "message": response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        context = {
            "description1": "Minsang's LLMREC",
            "description2": "준비중입니다.",
        }
        return render(request, "llmrec.html", context)



@csrf_exempt
def llmrec_soonhyeok(request):
    log_tracking(request=request, view='soonhyeok')
    username, session_id = get_username_sid(request, _from='llmrec/llmrec_soonhyeok GET')
    if request.method == 'POST':
        # Check if Neo4j GraphRAG_V2 is available
        if not NEO4J_GRAPHRAG_V2_AVAILABLE:
            return JsonResponse({
                'status': 'error',
                'message': '현재 이 기능은 사용할 수 없습니다. Neo4j 서버 연결이 필요합니다.'
            })

        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            print("message : ", message)
            query = message.get('text')
            print("query : ", query)
            log_llm(request=request, question=query, model_name='soonhyeok')

            # GraphState 생성 및 LangGraph 호출
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(recursion_limit=10, configurable={"thread_id": session_id})
            graph_state = Soonhyeok_GraphState(query=query)
            response_message = soonhyeok_app.invoke(graph_state, config=config)
            print("response_message : ", response_message)
            log_llm(request=request, answer=response_message, model_name='soonhyeok')

            # LangGraph 결과를 JSON 응답으로 반환
            if response_message.get("final_answer"):
                if response_message["query_type"] == 'general' :
                    return JsonResponse({
                        'status': 'success',
                        'query_type' : response_message['query_type'],
                        'message' : response_message["final_answer"],
                        'url' : '/llmrec/soonhyeok/'
                    })
                else :
                    # 영화 데이터를 가공하여 전달
                    movie_list = []
                    for movie in response_message["final_answer"]:
                        rating = movie.get('평점', {})
                        imdb_rating = rating.get('imdb', None)  # imdb 평점 가져오기
                        rotten_rating = rating.get('rotten tomatoes', None)  # Rotten Tomatoes 평점 가져오기

                        # 평점이 있는 경우만 추가
                        rating_info = []
                        if imdb_rating:
                            rating_info.append(f"IMDb {imdb_rating}")
                        if rotten_rating:
                            rating_info.append(f"Rotten {rotten_rating}")
                            
                        rating_text = ', '.join(rating_info) if rating_info else 'N/A'
                        
                        movie_list.append({
                            'title': movie.get('영화명', ''),
                            'poster': movie.get('poster', 'https://via.placeholder.com/150'),
                            'summary': movie.get('Synopsis Summary', ''),
                            'rating': rating_text,
                            'director': ', '.join(movie.get('감독', [])),
                            'actors': movie.get('출연 배우', ''),
                            'genre': movie.get('장르', ''),
                            'recommendation_reason': movie.get('추천 이유', ''),
                            'link': movie.get('movie_url', ['#']) if movie.get('movie_url') else {},
                        })
                    print("movie_list : ", movie_list)

                    return JsonResponse({
                        'status': 'success',
                        'movies': movie_list,
                        'query_type' : response_message['query_type'],
                        'url' : '/llmrec/soonhyeok/',
                        'intent' : response_message['intent']
                    })
            else:
                return JsonResponse({'status': 'error', 'message': "결과를 생성하지 못했습니다."})

        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Soonhyeok's LLMREC",
            'description2': "GrpahRAG을 기반으로 영화를 찾아 추천합니다. 최신 기술을 접목한 추천 챗봇을 사용해보세요!(샤라웃 경찬님👏)",
        }
        return render(request, "llmrec_soonhyeok.html", context)


@csrf_exempt
def llmrec_soonhyeok_Lite(request):
    log_tracking(request=request, view='soonhyeok_Lite')
    username, session_id = get_username_sid(request, _from='llmrec/llmrec_soonhyeok_Lite GET')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            print("message : ", message)
            query = message.get('text')
            print("query : ", query)
            log_llm(request=request, question=query, model_name='soonhyeok_Lite')

            # GraphState 생성 및 LangGraph 호출
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(recursion_limit=10, configurable={"thread_id": session_id})
            graph_state = Lite_GraphState(query=query)
            response_message = lite_app.invoke(graph_state, config=config)
            print("response_message : ", response_message)
            log_llm(request=request, answer=response_message, model_name='soonhyeok_Lite')

            # LangGraph 결과를 JSON 응답으로 반환
            if response_message.get("final_answer"):
                if response_message["query_type"] == 'general' :
                    return JsonResponse({
                        'status': 'success',
                        'query_type' : response_message['query_type'],
                        'message' : response_message["final_answer"],
                        'url' : '/llmrec/soonhyeok_Lite/'
                    })
                else :
                    # 영화 데이터를 가공하여 전달
                    movie_list = []
                    for movie in response_message["final_answer"]:
                        rating = movie.get('평점', {})
                        imdb_rating = rating.get('imdb', None)  # imdb 평점 가져오기
                        rotten_rating = rating.get('rotten tomatoes', None)  # Rotten Tomatoes 평점 가져오기

                        # 평점이 있는 경우만 추가
                        rating_info = []
                        if imdb_rating:
                            rating_info.append(f"IMDb {imdb_rating}")
                        if rotten_rating:
                            rating_info.append(f"Rotten {rotten_rating}")
                            
                        rating_text = ', '.join(rating_info) if rating_info else 'N/A'
                        
                        movie_list.append({
                            'title': movie.get('영화명', ''),
                            'poster': movie.get('poster', 'https://via.placeholder.com/150'),
                            'summary': movie.get('Synopsis Summary', ''),
                            'rating': rating_text,
                            'director': ', '.join(movie.get('감독', [])),
                            'actors': movie.get('출연 배우', ''),
                            'genre': movie.get('장르', ''),
                            'recommendation_reason': movie.get('추천 이유', ''),
                            'link': movie.get('movie_url', ['#']) if movie.get('movie_url') else {},
                        })
                    print("movie_list : ", movie_list)

                    return JsonResponse({
                        'status': 'success',
                        'movies': movie_list,
                        'query_type' : response_message['query_type'],
                        'url' : '/llmrec/soonhyeok_Lite/'
                    })
            else:
                return JsonResponse({'status': 'error', 'message': "결과를 생성하지 못했습니다."})

        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Soonhyeok's LLMREC",
            'description2': "Tavily 웹 검색을 기반으로 영화를 찾아 추천합니다. 최신 기술을 접목한 추천 챗봇을 사용해보세요!",
        }
        return render(request, "llmrec_soonhyeok.html", context)



@csrf_exempt
def llmrec_gyungah(request):
    log_tracking(request=request, view="gyungah")
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            message = data.get("message", "")

            question = message.get("text")
            log_llm(request=request, question=question, model_name="gyungah")
            new_response = g_get_chain(question)
            log_llm(request=request, answer=new_response, model_name="gyungah")

            print(
                f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}"
            )

            return JsonResponse(
                {
                    "status": "success",
                    "message": new_response,
                    "url": "/llmrec/gyungah/",
                }
            )
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        context = {
            "description1": "Gyungah's LLMREC",
            "description2": "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 장원영이에요!🎬✨ <br>귀엽고 긍정적인 말투로 여러분께 딱 맞는 영화를 추천해드릴게요! 🍿💖<br>GEMINI API를 사용했습니다!",
        }
        return render(request, "llmrec.html", context)