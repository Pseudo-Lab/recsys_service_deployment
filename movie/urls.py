from django.urls import path

from movie.views import home, movie_detail, search, sasrec, kprn, general_mf, ngcf

urlpatterns = [
    path("", home),
    path("movierecommendation/", home),
    path("<int:movie_id>", movie_detail),
    path("search/<str:keyword>/", search),
    path("sasrec/", sasrec),
    path("ngcf/", ngcf),
    path("kprn/", kprn),
    path("mf/", general_mf),
    # path("delete_all_history/", delete_all_history),
]
