from django.urls import path

from movie.views import home, delete_movie

urlpatterns = [
    path("movierec/", home),
    path('delete_movie/', delete_movie, name='delete_movie') 
]
