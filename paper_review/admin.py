from django.contrib import admin
from .models import Post
from markdownx.admin import MarkdownxModelAdmin
admin.site.register(Post, MarkdownxModelAdmin)

