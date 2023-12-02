from django.urls import path

from movie.views import home

urlpatterns = [
    path("movierec/", home),
    path("movierec/.pop", home),
    path("movierec/.sasrec", home)
]
