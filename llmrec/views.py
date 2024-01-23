from django.shortcuts import render

# Create your views here.
def llmrec(request):
    return render(request, 'llmrec.html')