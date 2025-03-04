from django import template
from paper_review.models import PostMonthlyPseudorec

register = template.Library()

@register.filter
def instanceof(obj, class_name):
    """템플릿에서 객체가 특정 클래스인지 확인하는 필터"""
    return isinstance(obj, PostMonthlyPseudorec)
