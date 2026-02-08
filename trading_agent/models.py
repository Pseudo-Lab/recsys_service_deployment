from django.db import models
from django.conf import settings


class InvestmentProfile(models.Model):
    """유저별 투자 프로필 저장 모델"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='investment_profile')

    # 투자 성향
    risk_tolerance = models.CharField(max_length=50, default='moderate')  # conservative, moderate, aggressive
    investment_horizon = models.CharField(max_length=50, default='medium-term')  # short-term, medium-term, long-term
    investment_style = models.CharField(max_length=50, blank=True, null=True)  # growth, value, income, balanced

    # 선호/비선호 섹터 (JSON 형태로 저장)
    preferred_sectors = models.JSONField(default=list, blank=True)
    avoided_sectors = models.JSONField(default=list, blank=True)

    # 커스텀 지시사항
    custom_instructions = models.TextField(blank=True, null=True)

    # 자유 입력 텍스트 (LLM 분석용)
    raw_text = models.TextField(blank=True, null=True)

    # 대화 히스토리 (JSON 형태로 저장)
    chat_history = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Investment Profile'
        verbose_name_plural = 'Investment Profiles'

    def __str__(self):
        return f"{self.user.username}'s Investment Profile"

    def to_dict(self):
        return {
            'risk_tolerance': self.risk_tolerance,
            'investment_horizon': self.investment_horizon,
            'investment_style': self.investment_style,
            'preferred_sectors': self.preferred_sectors,
            'avoided_sectors': self.avoided_sectors,
            'custom_instructions': self.custom_instructions,
            'raw_text': self.raw_text,
            'chat_history': self.chat_history,
        }


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
    news_analysis = models.TextField(blank=True)  # 뉴스 분석

    # 백테스트 정보
    start_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    end_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    start_date = models.CharField(max_length=20, blank=True)  # 분석일 실제 거래일
    end_date = models.CharField(max_length=20, blank=True)  # 5일 후 실제 거래일
    return_pct = models.CharField(max_length=20, blank=True)
    is_accurate = models.BooleanField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis History'
        verbose_name_plural = 'Analysis Histories'

    def __str__(self):
        return f"{self.user.username} - {self.ticker} ({self.analysis_date}) - {self.decision}"
