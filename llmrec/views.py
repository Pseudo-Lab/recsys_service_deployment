import json
import logging

import pandas as pd
import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from langchain.schema import HumanMessage

from clients import MysqlClient
from db_clients.dynamodb import DynamoDBClient
from llmrec.utils.gyungah.load_chain import get_chain as g_get_chain
from llmrec.utils.hyeonwoo.load_chain import router
from llmrec.utils.kyeongchan.get_model import kyeongchan_model
from llmrec.utils.log_questions import log_llm
from llmrec.utils.soonhyeok.GraphRAG import get_results
from movie.utils import get_username_sid, log_tracking, get_user_logs_df

mysql = MysqlClient()
load_dotenv('.env.dev')
table_llm = DynamoDBClient(table_name='llm')
table_clicklog = DynamoDBClient(table_name='clicklog')


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


@csrf_exempt
def llmrec_kyeongchan(request):
    log_tracking(request=request, view='kyeongchan')
    username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')
            question = message.get('text')
            log_llm(request=request, question=question, model_name='kyeongchan')

            # 여기서 message를 원하는 대로 처리
            print(f"[{message.get('timestamp')}]{username}({session_id}) : {message.get('text')}")

            ########################################## retrieval request
            # url = "http://3.36.208.188:8989/api/v1/retrieval/similarity_search/"
            #
            # payload = {
            #     "input": message.get('text'),
            #     "top_k": 2,
            #     "workspace_id": "76241726-616d-46bd-81ff-dfd07579d069"
            # }
            # headers = {
            #     "Content-Type": "application/json"
            # }
            # # response = requests.post(url, json=payload, headers=headers)
            # # context_extended = ', '.join([_[0]['page_content'] for _ in response.json()])
            #
            # template = '''You are an excellent movie curator. Your job is to recommend movie to user based on Context.
            # Context:
            #
            # Context : {}
            # Question: {input}
            #
            # Answer:'''
            # prompt_template = PromptTemplate.from_template(template)
            # chain = prompt_template | kyeongchan_model | StrOutputParser()
            # response_message = chain.invoke({"input": message.get('text')})
            # response_message = '아직 모델이 없어요..'
            ########################################## retrieval request

            response_message = kyeongchan_model([
                HumanMessage(message.get('text'))
            ])
            log_llm(request=request, answer=response_message, model_name='kyeongchan')
            print(f"response_message : {response_message}")

            def message_stream():
                yield json.dumps({'status': 'start', 'message': 'Streaming started...\n'})
                for content in response_message.content.split('. '):  # 예시로 문장 단위로 나누어 스트리밍
                    yield json.dumps({'status': 'stream', 'message': content + '.\n'})
                yield json.dumps({'status': 'end', 'message': 'Streaming ended.\n'})

            return StreamingHttpResponse(message_stream(), content_type='application/json')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            # return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        user_logs_df = get_user_logs_df(username, session_id)
        sorted_df = user_logs_df[['movieId', 'timestamp']].sort_values(by='timestamp')
        history_mids = []
        cnt = 0
        last_k = 10
        for mid in sorted_df['movieId']:
            if mid not in history_mids:
                history_mids.append(mid)
                cnt += 1
            if cnt == last_k:
                break

        sql = """
        SELECT *
        FROM daum_movies
        WHERE movieId in ({history_mids})
        """
        sql = sql.format(history_mids=', '.join([str(hmid) for hmid in history_mids]))
        with mysql.get_connection() as conn:
            history_df = pd.read_sql(sql, conn)

        preference_prompt = """다음은 유저가 최근 본 영화들이야. 이 영화들을 보고 유저의 영화 취향을 한 문장으로 설명해. 다른 말은 하지마.

        {history_with_newline}"""

        preference_response = kyeongchan_model([
            HumanMessage(preference_prompt)
        ])

        logging.info("로깅")

        # API 엔드포인트 URL
        url = 'http://127.0.0.1:7001/sasrec/'
        # 요청 헤더
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        # 요청 본문 데이터
        data = {
            "movie_ids": history_mids
        }
        # POST 요청 보내기
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # 봤던 영화 제거
        sasrec_recomm_mids = response.json()['sasrec_recomm_mids']
        sasrec_recomm_mids = [mid for mid in sasrec_recomm_mids if mid not in [int(_) for _ in history_mids]]

        sql = f"""
        SELECT *
        FROM daum_movies
        WHERE movieId IN ({','.join(map(str, sasrec_recomm_mids))})
        """
        with mysql.get_connection() as conn:
            df = pd.read_sql(sql, conn)
        df_sorted = df.set_index('movieId').loc[sasrec_recomm_mids].reset_index()

        profile = preference_response.content
        history_mtitles = ', '.join(history_df['titleKo'].tolist())
        candidates = ', '.join(df_sorted['titleKo'].tolist())

        recommendation_prompt = f"""너는 유능한 영화 평론가야.
        1. {username}님의 시청 이력에 기반해서 후보로부터 3가지 영화를 추천해.
        2. {username}님의 프로파일을 참고해. 추천 근거를 공손한 어투로 작성해줘. 

        출력 형식은 다음과 같아.
        [
            {{
                영화이름 : 추천 근거
            }},
            {{
                영화이름 : 추천 근거
            }},
            {{
                영화이름 : 추천 근거
            }},
        ]

        프로파일 : {profile}
        시청 이력 : {history_mtitles}
        후보 : {candidates}"""

        response_message = kyeongchan_model([
            HumanMessage(recommendation_prompt)
        ])
        recommendations = json.loads(response_message.content)
        answer_lines = []
        for recommendation in recommendations:
            for movie, reason in recommendation.items():
                answer_lines.append(f"{movie}: {reason}")
        initial_message = '<br>'.join(answer_lines)
        image = """
        <img src="https://img1.daumcdn.net/thumb/C408x596/?fname=https%3A%2F%2Ft1.daumcdn.net%2Fmovie%2F54d73561dce387c9a482cee6a47beacb6318d18e"
        alt="Daum Movie Image">
        """

        context = {
            'description1': "Kyeongchan's LLMREC",
            'description2': "Self-Querying을 이용한 RAG를 사용해 추천합니다!.",
            'initial_message': image + initial_message,
        }


        return render(request, "llmrec.html", context)


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
            'description2': "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 장원영이에요!🎬✨ <br>귀엽고 긍정적인 말투로 여러분께 딱 맞는 영화를 추천해드릴게요! 🍿💖"
        }
        return render(request, "llmrec.html", context)
