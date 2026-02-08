from django.urls import path
from . import views

urlpatterns = [
    path('', views.guiderec_home, name='guiderec_home'),
    path('chat/', views.guiderec_chat, name='guiderec_chat'),
]
