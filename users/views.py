from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect

# Create your views here.
from users.forms import LoginForm


def login_view(request):
    if request.user.is_authenticated:
        return redirect("/movie/movierec/")

    if request.method == "POST":
        form = LoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]

            user = authenticate(username=username, password=password)

            if user:
                login(request, user)
                return redirect("/movie/movierec")
            else:
                print(f"로그인 실패")

        context = {"form": form}
        return render(request, "users/login.html", context)
    else:
        form = LoginForm()
        context = {"form": form}
        return render(request, "users/login.html", context)


def logout_view(request):
    logout(request)
    return redirect("/users/login/")
