import json

import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from llmrec.utils import kyeongchan_model
from llmrec.utils.hyeonwoo.load_chain import get_chain
from movie.utils import get_username_sid


load_dotenv('.env.dev')


@csrf_exempt
def llmrec_hyeonwoo(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            question = message.get('text')
            new_response = get_chain(question)
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': new_response, 'url': '/llmrec/hyeonwoo/'})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Hyenwoo's LLMREC",
            'description2': "시놉시스 기반의 영화 추천입니다. 준비중입니다.",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_namjoon(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[남준님 모델]아직 모델이 없어요ㅠ'

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
    if request.method == 'POST':
        username, session_id = get_username_sid(request, _from='llmrec_kyeongchan')
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{username}({session_id}) : {message.get('text')}")

            # retrieval request ##########################
            url = "http://3.36.208.188:8989/api/v1/retrieval/similarity_search/"

            payload = {
                "input": message.get('text'),
                "top_k": 2,
                "workspace_id": "76241726-616d-46bd-81ff-dfd07579d069"
            }
            headers = {
                "Content-Type": "application/json"
            }
            # response = requests.post(url, json=payload, headers=headers)
            # context_extended = ', '.join([_[0]['page_content'] for _ in response.json()])

            template = '''You are an excellent movie curator. Your job is to recommend movie to user based on Context.
            Context:

            Context : {}
            Question: {input}
            
            Answer:'''
            prompt_template = PromptTemplate.from_template(template)
            chain = prompt_template | kyeongchan_model | StrOutputParser()
            response_message = chain.invoke({"input": message.get('text')})
            # response_message = '아직 모델이 없어요..'

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Kyeongchan's LLMREC",
            'description2': "준비중입니다.",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_minsang(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[민상님 모델]아직 모델이 없어요ㅠ'

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
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[순혁님 모델]아직 모델이 없어요ㅠ'

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
def llmrec_kwondong(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[권동님 모델]아직 모델이 없어요ㅠ'

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Kwondong's LLMREC",
            'description2': "준비중입니다.",
        }
        return render(request, "llmrec.html", context)


@csrf_exempt
def llmrec_gyungah(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            # TODO : 캐시로 히스토리 갖고있다가 multi-turn? 모델도 히스토리 모델이 필요하다. 한글, 챗, 히스토리 사용 가능한 모델이어야함.
            # TODO : 히스토리 어디 어떻게 저장?
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '[경아님 모델]아직 모델이 없어요ㅠ'

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        context = {
            'description1': "Kyeongah's LLMREC",
            'description2': "준비중입니다.",
        }
        return render(request, "llmrec.html", context)
