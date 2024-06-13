from django.urls import path

from llmrec.views import llmrec_hyeonwoo, llmrec_namjoon, llmrec_kyeongchan, llmrec_minsang, llmrec_soonhyeok, llmrec_gyungah

urlpatterns = [
    path("hyeonwoo/", llmrec_hyeonwoo),
    path("namjoon/", llmrec_namjoon),
    path("kyeongchan/", llmrec_kyeongchan),
    path("minsang/", llmrec_minsang),
    path("soonhyeok/", llmrec_soonhyeok),
    path("gyungah/", llmrec_gyungah),
]
