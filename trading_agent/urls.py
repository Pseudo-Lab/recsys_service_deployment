from django.urls import path
from . import views

app_name = 'trading_agent'

urlpatterns = [
    path('', views.trading_agent_index, name='index'),
    path('analyze/', views.analyze_stock, name='analyze'),
    path('api/profile/', views.user_profile_api, name='profile_api'),
    path('api/save_analysis/', views.save_analysis, name='save_analysis'),
    path('api/history/', views.get_analysis_history, name='get_history'),
    path('api/history/<int:analysis_id>/', views.get_analysis_detail, name='get_analysis_detail'),
    path('api/history/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
]
