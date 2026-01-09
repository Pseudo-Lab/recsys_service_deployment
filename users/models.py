from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.
class User(AbstractUser):
    profile_image = models.ImageField("프로필 이미지", upload_to="users/profile", blank=True, null=True)
    short_description = models.TextField("소개글", blank=True, null=True)
    phone_number = models.CharField("휴대폰 번호", max_length=20, blank=True, null=True)

