from django.contrib import admin
from .models import Post, PostMonthlyPseudorec, Comment
from markdownx.admin import MarkdownxModelAdmin


class PostAdmin(admin.ModelAdmin):
    save_on_top = True


admin.site.register(Post, MarkdownxModelAdmin)
admin.site.register(PostMonthlyPseudorec, MarkdownxModelAdmin)
admin.site.register(Comment)
