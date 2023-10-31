from django.shortcuts import redirect


def index(request):
    if request.user.is_authenticated:
        return redirect("/movie/movierec/")
    else:
        return redirect("/users/login/")
