from django.urls import path
from . import views

urlpatterns = [
    path('', views.my_personal_agents, name='my_agents'),
    path('api/fetch-news/', views.fetch_news_api, name='fetch_news_api'),
    # 기존 agent 관련 URL은 주석 처리 (나중에 필요하면 복구)
    # path('', views.my_agents_page, name='my_agents'),
    # path('<slug:agent_slug>/', views.my_agent_chat, name='my_agent_chat'),
]
