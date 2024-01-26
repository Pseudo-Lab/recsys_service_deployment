import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# @csrf_exempt
# def llmrec(request):
#     if request.method == 'POST':
#         data = json.loads(request.body.decode('utf-8'))
#         message = data.get('message', '')
#         print(f"data : {data}")
#         print(f"message.text : {message.get('text')}")
#     return render(request, 'llmrec.html')


@csrf_exempt
def llmrec(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            message = data.get('message', '')

            # 여기서 message를 원하는 대로 처리
            print(f"[{message.get('timestamp')}]{message.get('sender')} : {message.get('text')}")

            response_message = '아직 모델이 없어요ㅠ'

            # 클라이언트에게 성공적인 응답을 보냅니다.
            return JsonResponse({'status': 'success', 'message': response_message})
        except json.JSONDecodeError as e:
            # JSON 디코딩 오류가 발생한 경우 에러 응답을 보냅니다.
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return render(request, "llmrec.html")
