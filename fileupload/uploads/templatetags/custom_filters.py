from django import template
import re

register = template.Library()

@register.filter
def bold_text(value):
    if not isinstance(value, str):
        return value
    # Replace **text** with <strong>text</strong>
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', value)
