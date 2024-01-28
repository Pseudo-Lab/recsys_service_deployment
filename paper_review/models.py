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

    def set_content_from_md_file(self, md_file_path):
        # .md 파일에서 내용 읽어오기
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        self.content = md_content
        self.save()   # aws mysql로 바꾸면서 에러났는데 settings에서 'charset': 'utf8mb4' 추가하니 됐음
