from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect

from users.forms import LoginForm, SignupForm


def login_view(request):
    if request.user.is_authenticated:
        return redirect("/")

    if request.method == "POST":
        form = LoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]

            user = authenticate(username=username, password=password)

            if user:
                login(request, user)
                return redirect("/")
            else:
                form.add_error(None, "입력한 자격증명에 해당하는 사용자가 없습니다")

        context = {"form": form}
        return render(request, "users/login.html", context)
    else:
        form = LoginForm()
        context = {"form": form}
        return render(request, "users/login.html", context)


def logout_view(request):
    logout(request)
    return redirect("/")


def signup(request):
    if request.method == "POST":
        form = SignupForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            user = form.save()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            return redirect("/")
    else:  # GET 요청에서는 빈 Form을 보여준다.
        form = SignupForm()

    # context로 전달되는 form은 두 가지 경우가 존재
    # 1. POST 요청이며 form이 유효하지 않을 때 -> 에러를 포함한 form이 사용자에게 보여짐
    # 2. GET 요청일 때 : 빈 form이 사용자에게 보여짐
    context = {"form": form}
    return render(request, "users/signup.html", context)
