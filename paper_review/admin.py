from django.contrib import admin
from .models import Post, PostMonthlyPseudorec, Comment, PaperTalkPost
from markdownx.admin import MarkdownxModelAdmin


class PostAdmin(MarkdownxModelAdmin):
    save_on_top = True
    list_display = ('title', 'author', 'created_at', 'view_count')
    search_fields = ('title', 'content', 'author')

    # 필드셋으로 레이아웃 구성
    fieldsets = (
        (None, {
            'fields': (
                'title',
                'card_image',
                'content',
            ),
            'classes': ('wide',),
        }),
        ('작성자 정보', {
            'fields': (
                'author',
                'author_image',
                'author2',
                'author_image2',
            ),
        }),
        ('기타 정보', {
            'fields': (
                'view_count',
            ),
        }),
    )

    class Media:
        css = {
            'all': ('css/custom_admin.css',),  # 커스텀 CSS 추가
        }

class PostMonthlyPseudorecAdmin(MarkdownxModelAdmin):
    save_on_top = True
    list_display = ('title', 'month', 'tag1', 'tag2', 'view_count')
    search_fields = ('title', 'subtitle', 'tag1', 'tag2')

    fieldsets = (
        (None, {
            'fields': (
                'title',
                'subtitle',
                'month',
                'card_image',
                'content',
            ),
            'classes': ('wide',),
        }),
        ('작성자 정보', {
            'fields': (
                'author',
                'author_image',
            ),
        }),
        ('태그 정보', {
            'fields': (
                'tag1',
                'tag2',
            ),
        }),
        ('기타 정보', {
            'fields': (
                'view_count',
                'created_at',
            ),
        }),
    )

    class Media:
        css = {
            'all': ('css/custom_admin.css',),  # custom_admin.css를 나중에 추가
        }
        js = ('js/custom_admin.js',)



admin.site.register(Post, PostAdmin)
admin.site.register(PostMonthlyPseudorec, PostMonthlyPseudorecAdmin)
admin.site.register(Comment)


@admin.register(PaperTalkPost)
class PaperTalkPostAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'created_at', 'view_count')
    search_fields = ('title', 'author')
    list_filter = ('created_at',)
    ordering = ('-created_at',)