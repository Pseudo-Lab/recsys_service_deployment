from django.urls import path

from movie.views import home, movie_detail, search

urlpatterns = [
    path("movierec/", home),
    path("movierec/.pop", home),
    path("movierec/.sasrec", home),
    path("<int:movie_id>", movie_detail),
    path("search/<str:keyword>/", search),
]
