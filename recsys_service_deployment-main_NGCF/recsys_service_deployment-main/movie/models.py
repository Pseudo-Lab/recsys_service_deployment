from django.db import models


class WatchedMovie(models.Model):
    name = models.CharField(max_length=30)
