from django.db import models
from markdownx.utils import markdown
from markdownx.models import MarkdownxField

from users.models import User


class Post(models.Model):
    title = models.CharField(max_length=100)
    # content = models.TextField()
    card_image = models.ImageField("카드 이미지", upload_to="paper_review/card_imgs", blank=True)
    content = MarkdownxField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)
    author = models.CharField(max_length=50, default="작성자추가")
    author_image = models.ImageField("작성자 이미지", upload_to="paper_review/author_imgs", blank=True)
    author2 = models.CharField(max_length=50, default="작성자2추가")
    author_image2 = models.ImageField("작성자2 이미지", upload_to="paper_review/author_imgs", blank=True)
    view_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"[{self.pk}]{self.title}"

    def get_absolute_url(self):
        return f"{self.pk}"

    def get_content_markdown(self):
        return markdown(self.content)

    def set_content_from_md_file(self, md_file_path):
        # .md 파일에서 내용 읽어오기
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        self.content = md_content
        self.save()   # aws mysql로 바꾸면서 에러났는데 settings에서 'charset': 'utf8mb4' 추가하니 됐음

class PostMonthlyPseudorec(models.Model):
    title = models.CharField(max_length=100, default='제목 작성')
    subtitle = models.CharField(max_length=100, default='부제 작성')
    month = models.CharField(max_length=10, default='203004')
    card_image = models.ImageField("카드 이미지", upload_to="paper_review/card_imgs", blank=True)
    content = MarkdownxField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)
    author = models.CharField(max_length=50, default='작성자추가')
    author_image = models.ImageField("작성자 이미지", upload_to="paper_review/author_imgs", blank=True)
    tag1 = models.CharField(max_length=50, default='Recommendation Model')
    tag2 = models.CharField(max_length=50, default='Tech')
    view_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"[{self.pk}]{self.title}"

    def get_absolute_url(self):
        return f"{self.pk}"

    def get_content_markdown(self):
        return markdown(self.content)

    def set_content_from_md_file(self, md_file_path):
        # .md 파일에서 내용 읽어오기
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        self.content = md_content
        self.save()   # aws mysql로 바꾸면서 에러났는데 settings에서 'charset': 'utf8mb4' 추가하니 됐음

class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField()
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.author}::{self.content}"


class PaperTalkPost(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField(blank=True)  # 선택적으로 본문 내용 포함 가능
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    author = models.CharField(max_length=50, default="작성자추가")
    author_image = models.ImageField("작성자 이미지", upload_to="paper_review/author_imgs", blank=True)
    view_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"[{self.pk}] {self.title}"

    def get_absolute_url(self):
        return f"{self.pk}"
    
    
class PaperTalkComment(models.Model):
    post = models.ForeignKey(PaperTalkPost, on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.author.username}: {self.content[:30]}"  # 30글자까지만 표시