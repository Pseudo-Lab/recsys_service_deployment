from django.contrib import admin
from .models import Post, PostMonthlyPseudorec
from markdownx.admin import MarkdownxModelAdmin
admin.site.register(Post, MarkdownxModelAdmin)
admin.site.register(PostMonthlyPseudorec, MarkdownxModelAdmin)

