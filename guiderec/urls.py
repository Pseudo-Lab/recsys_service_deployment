from django.urls import path
from . import views

urlpatterns = [
    path('', views.guiderec_home, name='guiderec_home'),
    path('chat/', views.guiderec_chat, name='guiderec_chat'),

    # Session Management API
    path('api/sessions/', views.session_list, name='guiderec_session_list'),
    path('api/sessions/create/', views.session_create, name='guiderec_session_create'),
    path('api/sessions/<uuid:session_id>/', views.session_detail, name='guiderec_session_detail'),
    path('api/sessions/<uuid:session_id>/delete/', views.session_delete, name='guiderec_session_delete'),
]
