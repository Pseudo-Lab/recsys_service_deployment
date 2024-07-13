from django.urls import path

from llmrec.views import llmrec_hyeonwoo, llmrec_namjoon, llmrec_kyeongchan, llmrec_minsang, llmrec_soonhyeok, \
    llmrec_gyungah, get_initial_recommendation

urlpatterns = [
    path("hyeonwoo/", llmrec_hyeonwoo),
    path("namjoon/", llmrec_namjoon),
    path("kyeongchan/", llmrec_kyeongchan),
path('get_initial_recommendation/', get_initial_recommendation, name='get_initial_recommendation'),
    path("minsang/", llmrec_minsang),
    path("soonhyeok/", llmrec_soonhyeok),
    path("gyungah/", llmrec_gyungah),
    path("stream_chat", llmrec_kyeongchan),
]
