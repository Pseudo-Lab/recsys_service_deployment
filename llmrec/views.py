import json

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from langchain.schema import HumanMessage

# from llmrec.utils import kyeongchan_model
from db_clients.dynamodb import DynamoDBClient
from llmrec.utils.hyeonwoo.load_chain import router
from llmrec.utils.gyungah.load_chain import get_chain as g_get_chain
from llmrec.utils.kyeongchan.get_model import kyeongchan_model
from llmrec.utils.log_questions import log_llm
from movie.utils import get_username_sid, log_tracking

load_dotenv('.env.dev')
table_llm = DynamoDBClient(table_name='llm')


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
            'description1': "Hyenwoo's LLMREC",
            'description2': "스마트한 영화 선택, LLM 기반의 영화 추천 서비스로 시작하세요!",
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
            return JsonResponse({'status': 'success', 'message': response_message})
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
    if request.method == 'POST':
        username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
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
        context = {
            'description1': "Kyeongchan's LLMREC",
            'description2': "Self-Querying을 이용한 RAG를 사용해 추천합니다!.",
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

            response_message = '[순혁님 모델]아직 모델이 없어요ㅠ'
            log_llm(request=request, answer=response_message, model_name='soonhyeok')

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Soonhyeok's LLMREC",
            'description2': "준비중입니다.",
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

            return JsonResponse({'status': 'success', 'message': new_response})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Kyeongah's LLMREC",
            'description2': "안녕하세요! 저는 PseudoRec에서 개발된 영화 추천 AI 장원영이에요!🎬✨ <br>귀엽고 긍정적인 말투로 여러분께 딱 맞는 영화를 추천해드릴게요! 🍿💖"
        }
        return render(request, "llmrec.html", context)
