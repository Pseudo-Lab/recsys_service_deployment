from django import template
from paper_review.models import Post, PostMonthlyPseudorec
import re

register = template.Library()


@register.filter
def instanceof(obj, class_name):
    """템플릿에서 객체가 특정 클래스인지 확인하는 필터"""
    return isinstance(obj, PostMonthlyPseudorec)


@register.filter
def isinstance_post(obj, _):
    """Post 모델 인스턴스인지 확인"""
    return isinstance(obj, Post)


@register.filter
def is_absolute_url(value):
    """주어진 값이 절대 URL인지 확인하는 필터"""
    return bool(re.match(r"^https?://", str(value)))


@register.filter
def image_url(value):
    """ImageField 또는 URLField 값을 안전하게 URL로 변환"""
    if not value:
        return ''
    if hasattr(value, 'url'):
        return value.url
    url = str(value)
    if url.startswith('http'):
        return url
    return f'/media/{url}'
