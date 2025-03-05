from django import template
from paper_review.models import PostMonthlyPseudorec
import re

register = template.Library()


@register.filter
def instanceof(obj, class_name):
    """템플릿에서 객체가 특정 클래스인지 확인하는 필터"""
    return isinstance(obj, PostMonthlyPseudorec)


@register.filter
def is_absolute_url(value):
    """주어진 값이 절대 URL인지 확인하는 필터"""
    return bool(re.match(r"^https?://", str(value)))
