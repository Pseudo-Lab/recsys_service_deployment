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

from movie.views import home, log_click, log_star

urlpatterns = [
    path("admin/", admin.site.urls),
    path("users/", include("users.urls")),
    path("movie/", include("movie.urls")),
    path('', home),
    path('log_click/', log_click, name='log_click'),
    path('log_star/', log_star, name='log_star'),
    path('paper_review/', include('paper_review.urls')),
    path('markdownx/', include('paper_review.urls')),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
