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
<<<<<<< HEAD
from django.urls import path
from config.views import home
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('log_click/', views.log_click, name='log_click')
=======
from django.urls import path, include

from config.views import index

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", index),
    path("users/", include("users.urls")),
    path("movie/", include("movie.urls"))
>>>>>>> origin/포스터-클릭-왼쪽에-리스팅하기
]
urlpatterns += static(
    prefix=settings.STATIC_URL,
    document_root=settings.STATIC_ROOT)
