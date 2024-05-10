from django.urls import path
from . import views

urlpatterns = [
    # path('<int:pk>/', views.PostDetail.as_view()),
    # path('', views.PostList.as_view()),
    path('paper_review/<int:pk>/', views.single_post_page_paper_review),
    path('paper_review/', views.index_paper_review),
    path('monthly_pseudorec/', views.index_monthly_pseudorec),
    path('monthly_pseudorec/<int:pk>/', views.single_post_page_monthly_pseudorec)
]
