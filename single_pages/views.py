from django.shortcuts import render

# Create your views here.
def about_us(request):
    return render(
        request,
        'single_pages/about_us.html'
    )

def trading_agent(request):
    return render(
        request,
        'single_pages/trading_agent.html'
    )