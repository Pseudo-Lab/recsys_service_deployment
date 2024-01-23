from django.urls import path

from llmrec.views import llmrec

urlpatterns = [
    path("", llmrec),
]
