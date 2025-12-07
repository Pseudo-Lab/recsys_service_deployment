from django.urls import path

from movie.views import home, movie_detail, search, sasrec, kprn, general_mf, ngcf, delete_movie_interaction, \
    delete_all_interactions, movie_recommendation_home

urlpatterns = [
    path("", home),
    path("movierecommendation/", movie_recommendation_home),
    path("<int:movie_id>", movie_detail),
    path("search/<str:keyword>/", search),
    path("sasrec/", sasrec),
    path("ngcf/", ngcf),
    path("kprn/", kprn),
    path("mf/", general_mf),
    path('delete_movie_interaction/', delete_movie_interaction, name='delete_movie_interaction'),
    path('delete_all_interactions/', delete_all_interactions, name='delete_all_interactions')

]
