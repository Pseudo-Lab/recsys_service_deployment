from django.contrib import admin
from movie.models import WatchedMovie

@admin.register(WatchedMovie)
class MovieAdmin(admin.ModelAdmin):
    pass
