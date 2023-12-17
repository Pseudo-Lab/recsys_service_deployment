from django.shortcuts import render


def index(request):
    return render(request, 'paper_review/index.html')
