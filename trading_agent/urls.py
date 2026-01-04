from django.urls import path
from . import views

app_name = 'trading_agent'

urlpatterns = [
    path('', views.trading_agent_index, name='index'),
    path('analyze/', views.analyze_stock, name='analyze'),
    path('api/profile/', views.user_profile_api, name='profile_api'),
]
