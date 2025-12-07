from django.urls import path

from llmrec.views import llmrec_hyeonwoo, llmrec_namjoon, llmrec_kyeongchan, llmrec_minsang, llmrec_soonhyeok, llmrec_soonhyeok_Lite, llmrec_gyungah, pplrec
from movie.views import llmrec_home


urlpatterns = [
    path("", llmrec_home, name='llmrec_home'),  # LLM Recommendation Demos 홈 페이지
    path("pplrec/", pplrec),
    path("hyeonwoo/", llmrec_hyeonwoo),
    path("namjoon/", llmrec_namjoon),
    path("kyeongchan/", llmrec_kyeongchan),
    # path('get_initial_recommendation/', get_initial_recommendation, name='get_initial_recommendation'),
    path("minsang/", llmrec_minsang),
    path("soonhyeok/", llmrec_soonhyeok, name='llmrec_soonhyeok'),
    path("soonhyeok_Lite/", llmrec_soonhyeok_Lite, name='llmrec_soonhyeok_Lite'),
    path("gyungah/", llmrec_gyungah),
    path("stream_chat", llmrec_kyeongchan),
]
