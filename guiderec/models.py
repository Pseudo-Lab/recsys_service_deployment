import uuid
from django.db import models
from django.conf import settings


class ChatSession(models.Model):
    """채팅 세션 (대화방)"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='guiderec_sessions',
        null=True,
        blank=True
    )
    title = models.CharField(max_length=200, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-updated_at']
        db_table = 'guiderec_chat_session'

    def __str__(self):
        return f"{self.title} ({self.id})"

    def generate_title_from_first_message(self):
        """첫 번째 사용자 메시지로 제목 생성"""
        first_message = self.messages.filter(role='user').first()
        if first_message:
            content = first_message.content[:50]
            self.title = content + "..." if len(first_message.content) > 50 else content
            self.save(update_fields=['title'])


class ChatMessage(models.Model):
    """채팅 메시지"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        db_table = 'guiderec_chat_message'

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
