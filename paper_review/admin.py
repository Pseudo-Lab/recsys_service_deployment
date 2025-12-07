from django.contrib import admin
from .models import PaperTalkComment, Post, PostMonthlyPseudorec, Comment, PaperTalkPost
from markdownx.admin import MarkdownxModelAdmin


class PostAdmin(MarkdownxModelAdmin):
    save_on_top = True
    list_display = ('title', 'category', 'subcategory', 'author', 'created_at', 'view_count')
    search_fields = ('title', 'content', 'author', 'category', 'subcategory')
    list_filter = ('category', 'subcategory')

    # 필드셋으로 레이아웃 구성
    fieldsets = (
        (None, {
            'fields': (
                'title',
                'category',
                'subcategory',
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
    list_display = ('title', 'category', 'subcategory', 'month', 'author', 'tag1', 'tag2', 'view_count', 'card_image_preview')
    list_display_links = ('title',)
    search_fields = ('title', 'subtitle', 'tag1', 'tag2', 'author', 'category', 'subcategory')
    list_filter = ('month', 'author', 'tag1', 'category', 'subcategory')
    readonly_fields = ('card_image_preview', 'author_image_preview')

    fieldsets = (
        (None, {
            'fields': (
                'title',
                'subtitle',
                'month',
                'category',
                'subcategory',
                'card_image',
                'card_image_preview',
                'content',
            ),
            'classes': ('wide',),
        }),
        ('작성자 정보', {
            'fields': (
                'author',
                'author_image',
                'author_image_preview',
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

    def card_image_preview(self, obj):
        if obj.card_image:
            return f'<img src="{obj.card_image.url}" style="max-height: 200px; max-width: 300px;" />'
        return "이미지 없음"
    card_image_preview.short_description = '카드 이미지 미리보기'
    card_image_preview.allow_tags = True

    def author_image_preview(self, obj):
        if obj.author_image:
            return f'<img src="{obj.author_image.url}" style="max-height: 100px; max-width: 100px; border-radius: 50%;" />'
        return "이미지 없음"
    author_image_preview.short_description = '작성자 이미지 미리보기'
    author_image_preview.allow_tags = True

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

@admin.register(PaperTalkComment)
class PaperTalkCommentAdmin(admin.ModelAdmin):
    list_display = ('post', 'author', 'content', 'created_at')
    search_fields = ('author__username', 'content')
    list_filter = ('created_at',)
    ordering = ('-created_at',)
