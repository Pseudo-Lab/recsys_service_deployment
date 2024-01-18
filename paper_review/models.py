from django.db import models
from markdownx.utils import markdown
from markdownx.models import MarkdownxField


class Post(models.Model):
    title = models.CharField(max_length=30)
    # content = models.TextField()
    content = MarkdownxField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"[{self.pk}]{self.title}"

    def get_absolute_url(self):
        return f"/paper_review/{self.pk}"

    def get_content_markdown(self):
        return markdown(self.content)
