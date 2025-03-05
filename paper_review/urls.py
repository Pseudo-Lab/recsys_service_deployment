from django.urls import path
from . import views

urlpatterns = [
    # path('<int:pk>/', views.PostDetail.as_view()),
    # path('', views.PostList.as_view()),
    path(
        "paper_review/<int:pk>/",
        views.single_post_page_paper_review,
        name="single_post_page_paper_review",
    ),
    path("paper_review/", views.index_paper_review),
    path("paper_talk/", views.index_paper_talk, name='index_paper_talk'),
    path("paper_talk/add/", views.add_paper_talk_post, name="add_paper_talk_post"),
    path("paper_talk/<int:post_id>/edit/", views.edit_paper_talk_post, name="edit_paper_talk_post"),  # ✅ 추가
    path('paper_talk/<int:post_id>/add_comment/', views.add_paper_talk_comment, name='add_paper_talk_comment'),
    path('paper_talk/comment/<int:comment_id>/edit/', views.edit_paper_talk_comment, name='edit_paper_talk_comment'),
    path('paper_talk/comment/<int:comment_id>/delete/', views.delete_paper_talk_comment, name='delete_paper_talk_comment'),
    
    # ✅ 월간슈도렉 관련 URL
    path("monthly_pseudorec/", views.index_monthly_pseudorec, name="index_monthly_pseudorec"),
    path("monthly_pseudorec/<int:pk>/", views.single_post_page_monthly_pseudorec, name="single_post_page_monthly_pseudorec"),
    path("monthly_pseudorec/add/", views.add_monthly_pseudorec_post, name="add_monthly_pseudorec_post"),
    path("monthly_pseudorec/<int:pk>/edit/", views.edit_monthly_pseudorec_post, name="edit_monthly_pseudorec_post"),
    path("upload_image_ajax/", views.upload_image_ajax, name="upload_image_ajax"),
    path("monthly_pseudorec/<int:pk>/delete/", views.delete_monthly_pseudorec_post, name="delete_monthly_pseudorec_post"),

    # ✅ Paper Talk 관련 URL
    path("comment/edit/<int:comment_id>/", views.edit_comment, name="edit_comment"),
    path("comment/delete/<int:comment_id>/", views.delete_comment, name="delete_comment"),
    path("post_preview/", views.post_preview, name="post_preview"),
]
