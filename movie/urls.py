from django.urls import path

from movie.views import home

urlpatterns = [
    path("movierec/", home),
]
