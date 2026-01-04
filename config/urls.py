"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from movie.views import log_click, log_star
from my_agents.views import study_archive_home

urlpatterns = [
    path("admin/", admin.site.urls),
    path("users/", include("users.urls")),
    path("movie/", include("movie.urls")),
    path('', study_archive_home, name='home'),  # 첫 화면: STUDY ARCHIVE
    path('log_click/', log_click, name='log_click'),
    path('log_star/', log_star, name='log_star'),
    path('archive/', include('paper_review.urls')),
    # path('paper_review/', include('paper_review.urls')),
    # path('monthly_pseudorec/', include('paper_review.urls')),
    path('markdownx/', include('markdownx.urls')),
    path('about_us/', include('single_pages.urls')),
    path('trading_agent/', include('trading_agent.urls')),  # Trading Agent 앱
    # path('llmrec/', include('llmrec.urls')),  # TODO: Fix langchain compatibility
    path('accounts/', include('allauth.urls')),
    # path('sasrec/', include('movie.urls')),
    path('ngcf/', include("movie.urls")),
    path('movie_recommendation/', include("movie.urls")),  # ML/DL 영화 추천
    path('my_agents/', include('my_agents.urls')),  # MY AGENTS 페이지

]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
