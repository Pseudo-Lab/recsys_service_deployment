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
from llmrec.utils.kyeongchan.langgraph_test import app, GraphState
from llmrec.utils.kyeongchan.utils import get_landing_page_recommendation
from llmrec.utils.kyeongchan.search_engine import SearchManager
from llmrec.utils.log_questions import log_llm
from llmrec.utils.soonhyeok.GraphRAG import get_results
from movie.utils import get_username_sid, log_tracking, get_user_logs_df, get_interacted_movie_dicts

mysql = MysqlClient()
load_dotenv('.env.dev')
table_llm = DynamoDBClient(table_name='llm')
table_clicklog = DynamoDBClient(table_name='clicklog')


@csrf_exempt
def pplrec(request):
    log_tracking(request=request, view='pplrec')
    username, session_id = get_username_sid(request, _from='llmrec/pplrec GET')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            question = message.get('text')
            log_llm(request=request, question=question, model_name='pplrec')
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")
            new_response = '준비중이예요오오오'
            log_llm(request=request, answer=new_response, model_name='pplrec')

            return JsonResponse({'status': 'success', 'message': new_response})
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        user_logs_df = get_user_logs_df(username, session_id)
        interacted_movie_d = get_interacted_movie_dicts(user_logs_df)
        context = {
            'description1': "Pseudorec's Personalized LLM Recommendation",
            'description2': "슈도렉 멤버가 다같이 만드는 영화 A-Z LLM 모델",
            'watched_movie': interacted_movie_d,
        }
        return render(request, "llmrec_pplrec.html", context)

@csrf_exempt
def llmrec_hyeonwoo(request):
    log_tracking(request=request, view='hyeonwoo')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            question = message.get('text')
            log_llm(request=request, question=question, model_name='hyeonwoo')
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")
            new_response = router(question)
            log_llm(request=request, answer=new_response, model_name='hyeonwoo')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': new_response, 'url': '/llmrec/hyeonwoo/'})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Hyeonwoo's LLMREC",
            'description2': "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 코난이에요!🎬✨ <br>명탐정으로서 여러분들의 요구사항을 해결할게요 🕵️"
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_namjoon(request):
    log_tracking(request=request, view='namjoon')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            question = message.get('text')
            log_llm(request=request, question=question, model_name='namjoon')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[남준님 모델]아직 모델이 없어요ㅠ'
            log_llm(request=request, answer=response_message, model_name='namjoon')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message, 'url': '/llmrec/namjoon/'})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Namjoon's LLMREC",
            'description2': "남준님의 모델소개 : 준비중입니다!",
        }
        return render(request, "llmrec.html", context)


# @csrf_exempt
# def llmrec_kyeongchan.html(request):
#     log_tracking(request=request, view='kyeongchan')
#     username, session_id = get_username_sid(request, _from='llmrec_kyeongchan.html')
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body.decode('utf-8'))
#             message = data.get('message', '')
#             question = message.get('text')
#             log_llm(request=request, question=question, model_name='kyeongchan')
#
#             # 여기서 message를 원하는 대로 처리
#             print(f"[{message.get('timestamp')}]{username}({session_id}) : {message.get('text')}")
#
#             ########################################## retrieval request
#             # url = "http://3.36.208.188:8989/api/v1/retrieval/similarity_search/"
#             #
#             # payload = {
#             #     "input": message.get('text'),
#             #     "top_k": 2,
#             #     "workspace_id": "76241726-616d-46bd-81ff-dfd07579d069"
#             # }
#             # headers = {
#             #     "Content-Type": "application/json"
#             # }
#             # # response = requests.post(url, json=payload, headers=headers)
#             # # context_extended = ', '.join([_[0]['page_content'] for _ in response.json()])
#             #
#             # template = '''You are an excellent movie curator. Your job is to recommend movie to user based on Context.
#             # Context:
#             #
#             # Context : {}
#             # Question: {input}
#             #
#             # Answer:'''
#             # prompt_template = PromptTemplate.from_template(template)
#             # chain = prompt_template | kyeongchan_model | StrOutputParser()
#             # response_message = chain.invoke({"input": message.get('text')})
#             # response_message = '아직 모델이 없어요..'
#             ########################################## retrieval request
#
#             response_message = kyeongchan_model([
#                 HumanMessage(message.get('text'))
#             ])
#             log_llm(request=request, answer=response_message, model_name='kyeongchan')
#             print(f"response_message : {response_message}")
#
#             def message_stream():
#                 yield json.dumps({'status': 'start', 'message': 'Streaming started...\n'})
#                 for content in response_message.content.split('. '):  # 예시로 문장 단위로 나누어 스트리밍
#                     yield json.dumps({'status': 'stream', 'message': content + '.\n'})
#                 yield json.dumps({'status': 'end', 'message': 'Streaming ended.\n'})
#
#             return StreamingHttpResponse(message_stream(), content_type='application/json')
#
#             # 클라이언트에게 성공적인 응답을 보냅니다.
#             # return JsonResponse({'status': 'success', 'message': response_message})
#         except json.JSONDecodeError as e:
#             # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
#             return JsonResponse({'status': 'error', 'message': str(e)})
#     else:
#         user_logs_df = get_user_logs_df(username, session_id)
#         sorted_df = user_logs_df[['movieId', 'timestamp']].sort_values(by='timestamp')
#         history_mids = []
#         cnt = 0
#         last_k = 10
#         for mid in sorted_df['movieId']:
#             if mid not in history_mids:
#                 history_mids.append(mid)
#                 cnt += 1
#             if cnt == last_k:
#                 break
#
#         sql = """
#         SELECT *
#         FROM daum_movies
#         WHERE movieId in ({history_mids})
#         """
#         sql = sql.format(history_mids=', '.join([str(hmid) for hmid in history_mids]))
#         history_df = pd.read_sql(sql, mysql.engine)
#
#         preference_prompt = """다음은 유저가 최근 본 영화들이야. 이 영화들을 보고 유저의 영화 취향을 한 문장으로 설명해. 다른 말은 하지마.
#
#         {history_with_newline}"""
#
#         preference_response = kyeongchan_model([
#             HumanMessage(preference_prompt)
#         ])
#
#         logger.info("로깅")
#
#         url = 'http://127.0.0.1:7001/sasrec/'
#         # 요청 헤더
#         headers = {
#             'accept': 'application/json',
#             'Content-Type': 'application/json'
#         }
#         # 요청 본문 데이터
#         data = {
#             "movie_ids": history_mids
#         }
#         # POST 요청 보내기
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         # 봤던 영화 제거
#         sasrec_recomm_mids = response.json()['sasrec_recomm_mids']
#         sasrec_recomm_mids = [mid for mid in sasrec_recomm_mids if mid not in [int(_) for _ in history_mids]]
#
#         sql = f"""
#         SELECT *
#         FROM daum_movies
#         WHERE movieId IN ({','.join(map(str, sasrec_recomm_mids))})
#         """
#         df = pd.read_sql(sql, mysql.engine)
#         df_sorted = df.set_index('movieId').loc[sasrec_recomm_mids].reset_index()
#
#         candidates_lst = []
#         for _, row in df_sorted[['movieId', 'titleKo']].iterrows():
#             candidates_lst.append(f"{row['titleKo']}({row['movieId']})")
#
#         profile = preference_response.content
#         history_mtitles = ', '.join(history_df['titleKo'].tolist())
#         candidates = ', '.join(candidates_lst)
#
#         recommendation_prompt = f"""너는 유능하고 친절한 영화 전문가이고 영화 추천에 탁월한 능력을 갖고 있어. 너의 작업은 :
#         1. {username}님에게 후보로부터 1가지 영화를 골라 추천해줘.
#         2. 시청한 영화들의 특징을 꼼꼼히 분석해서 타당한 추천 근거를 들어줘. 장르, 스토리, 인기도, 감독, 배우 등을 분석하면 좋아.
#         3. 추천 근거를 정성스럽고 길게 작성해줘.
#
#         출력 형식은 다음과 같이 json으로 반환해줘.
#
#         {{
#             "titleKo" : "영화 이름1",
#             "movieId" : "영화 id",
#             "reason" : "추천 근거"
#         }}
#
#         시청 이력 : {history_mtitles}
#         후보 : {candidates}"""
#
#         response_message = kyeongchan_model([
#             HumanMessage(recommendation_prompt)
#         ])
#         recommendations = json.loads(response_message.content)
#         recommended_mid = int(recommendations['movieId'])
#         sql = f"""
#         SELECT dm.movieId,
#         dm.posterUrl,
#         dmsp.synopsis_prep
#         FROM daum_movies dm
#         LEFT JOIN daum_movies_synopsis_prep dmsp ON dm.movieId = dmsp.movieId
#         where dm.movieId = {recommended_mid}
#         """
#         df = pd.read_sql(sql, mysql.engine)
#         poster_url = df.iloc[0]['posterUrl']
#         synopsis_prep = df.iloc[0]['synopsis_prep']
#
#         image = f"""
#         <img src="{poster_url}" alt="Daum Movie Image" style="width: 300px;">
#         """
#
#         answer = image + '<br>' + recommendations['reason'] + '<br><bn>시놉시스<br>' + synopsis_prep
#
#         context = {
#             'description1': "Kyeongchan's LLMREC",
#             'description2': "Self-Querying을 이용한 RAG를 사용해 추천합니다!.",
#             'initial_message': answer,
#         }
#
#         return render(request, "llmrec.html", context)

# @csrf_exempt
# def llmrec_kyeongchan(request):
#     log_tracking(request=request, view='kyeongchan')
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body.decode('utf-8'))
#             message = data.get('message', '')
#             question = message.get('text')
#             log_llm(request=request, question=question, model_name='kyeongchan')
#
#             print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")
#
#             response_message = '[경찬님 모델]아직 모델이 없어요ㅠ'
#             log_llm(request=request, answer=response_message, model_name='minsang')
#
#             # 클라이언트에게 성공적인 응답을 보냅니다.
#             return JsonResponse({'status': 'success', 'message': response_message})
#         except json.JSONDecodeError as e:
#             # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
#             return JsonResponse({'status': 'error', 'message': str(e)})
#     else:
#         context = {
#             'description1': "Kyeongchan's LLMREC",
#             'description2': "준비중입니다.",
#         }
#         return render(request, "llmrec.html", context)
# @csrf_exempt
# def stream_chat(request):
#     text = request.GET.get('text')
    # search_manager = SearchManager(
    #     api_key="",
    #     index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
    #     top_k=5,
    #     score_threshold=0.7
    # )
    # search_manager.add_engine("self")
    # # context = search_manager.search_all(text)
    # template = '''You are an excellent movie curator. Your job is to recommend movie to user based on Context.
    # Context:
    # {context}
    #
    # Question: {input}
    # Answer:'''
    #
    # prompt_template = PromptTemplate.from_template(template)
    # chain = prompt_template | kyeongchan_model | StrOutputParser()

    # def message_stream():
    #     for chunk in chain.stream({"input": text, "context": "봉준호 감독 영화"}):
    #         data = json.dumps(chunk)
    #         yield f'data: {data}\n\n'



    # return StreamingHttpResponse(message_stream(), content_type='text/event-stream')


# @csrf_exempt
# def llmrec_kyeongchan(request):
#     username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
#     log_tracking(request=request, view='kyeongchan')
#     if request.method == 'GET':
#         user_logs_df = get_user_logs_df(username, session_id)
#         if len(user_logs_df):
#             print(user_logs_df)
#             answer = get_landing_page_recommendation(username, user_logs_df, kyeongchan_model)
#             context = {
#                 'description1': "Kyeongchan & Byeongcheol LLMREC",
#                 'description2': "Self-Querying을 이용한 RAG를 사용해 추천합니다!.",
#                 'initial_message': answer,
#             }
#         else:
#             context = {
#                 'description1': "Kyeongchan & Byeongcheol LLMREC",
#                 'description2': "Self-Querying을 이용한 RAG를 사용해 추천합니다!.",
#                 'initial_message': '',
#             }
#
#         return render(request, "llmrec_kyeongchan.html", context)


from django.http import JsonResponse

@csrf_exempt
def llmrec_kyeongchan(request):
    username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
    if request.method == 'GET':
        log_tracking(request=request, view='kyeongchan')
        user_logs_df = get_user_logs_df(username, session_id)
        interacted_movie_d = get_interacted_movie_dicts(user_logs_df)
        context = {
            'description1': "깃잔심팀 - LLM 영화 개인화 추천",
            # 'description2': "Self-Querying RAG 기법을 사용한 추천",
            # 'initial_message': f'{username}의 취향을 분석 중입니다...',
            'initial_message': "홈화면에서 좋아하는 영화 평점을 매긴 후, '봉준호 감독이 연출한 영화 추천해줘'와 같은 질문을 해보세요!<br>GEMINI API를 사용했습니다!",
            'watched_movie': interacted_movie_d
        }
        return render(request, "llmrec_kyeongchan.html", context)
    else:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('message', '')['text']
        from langchain_core.runnables import RunnableConfig
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "movie"})
        inputs = GraphState(question=message, username=username)
        response_message = app.invoke(inputs, config=config)
        return JsonResponse({'status': 'success', 'message': response_message['final_answer']})



# @csrf_exempt
# def get_initial_recommendation(request):
#     username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
#     user_logs_df = get_user_logs_df(username, session_id)
#     if len(user_logs_df):
#         answer = get_landing_page_recommendation(username, user_logs_df, kyeongchan_model)
#     else:
#         answer = ''
#     print(f"answer : {answer}")
#     return JsonResponse({'status': 'success', 'message': answer})


@csrf_exempt
def llmrec_minsang(request):
    log_tracking(request=request, view='minsang')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            question = message.get('text')
            log_llm(request=request, question=question, model_name='minsang')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[민상님 모델]아직 모델이 없어요ㅠ'
            log_llm(request=request, answer=response_message, model_name='minsang')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Minsang's LLMREC",
            'description2': "준비중입니다.",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_soonhyeok(request):
    log_tracking(request=request, view='soonhyeok')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            question = message.get('text')
            log_llm(request=request, question=question, model_name='soonhyeok')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = get_results(question)
            log_llm(request=request, answer=response_message, model_name='soonhyeok')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Soonhyeok's LLMREC",
            'description2': "GrpahRAG을 기반으로 유사한 영화를 찾아 추천합니다. 최신 기술을 접목한 추천 챗봇을 사용해보세요!",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_gyungah(request):
    log_tracking(request=request, view='gyungah')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            question = message.get('text')
            log_llm(request=request, question=question, model_name='gyungah')
            new_response = g_get_chain(question)
            log_llm(request=request, answer=new_response, model_name='gyungah')

            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            return JsonResponse({'status': 'success', 'message': new_response, 'url': '/llmrec/gyungah/'})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Gyungah's LLMREC",
            'description2': "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 장원영이에요!🎬✨ <br>귀엽고 긍정적인 말투로 여러분께 딱 맞는 영화를 추천해드릴게요! 🍿💖<br>GEMINI API를 사용했습니다!"
        }
        return render(request, "llmrec.html", context)
