from django.db import models
from django.conf import settings


class ProfileChatHistory(models.Model):
    """대화 내역 저장 모델"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile_chats')
    message = models.TextField()
    is_user_message = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Profile Chat History'
        verbose_name_plural = 'Profile Chat Histories'

    def __str__(self):
        msg_type = "User" if self.is_user_message else "System"
        return f"{self.user.username} - {msg_type}: {self.message[:50]}..."


class AnalysisHistory(models.Model):
    """분석 결과 저장 모델"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='analysis_history')
    ticker = models.CharField(max_length=10)
    analysis_date = models.DateField()
    decision = models.CharField(max_length=50)

    # 분석 결과 저장
    investment_plan = models.TextField()
    risk_assessment = models.TextField(blank=True)
    technical_analysis = models.TextField(blank=True)
    fundamental_analysis = models.TextField(blank=True)
    sentiment_analysis = models.TextField(blank=True)

    # 백테스트 정보
    start_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    end_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    return_pct = models.CharField(max_length=20, blank=True)
    is_accurate = models.BooleanField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis History'
        verbose_name_plural = 'Analysis Histories'

    def __str__(self):
        return f"{self.user.username} - {self.ticker} ({self.analysis_date}) - {self.decision}"
