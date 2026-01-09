from django import forms
from django.core.exceptions import ValidationError

from users.models import User


class LoginForm(forms.Form):
    username = forms.CharField(min_length=3, widget=forms.TextInput(attrs={"placeholder": "아이디 (3자리 이상)"}), label='아이디')
    password = forms.CharField(min_length=3, widget=forms.PasswordInput(attrs={"placeholder": "비밀번호 (4자리 이상)"}), label='비밀번호')


class SignupForm(forms.Form):
    username = forms.CharField(label='아이디')
    email = forms.EmailField(label='이메일', required=True)
    password1 = forms.CharField(widget=forms.PasswordInput, label='비밀번호')
    password2 = forms.CharField(widget=forms.PasswordInput, label='비밀번호 확인')
    phone_number = forms.CharField(required=False, max_length=20, label='휴대폰 번호 (선택)')
    profile_image = forms.ImageField(required=False, label='프로필 이미지 (선택)')
    short_description = forms.CharField(required=False, label='소개글 (선택)')

    def clean_username(self):
        username = self.cleaned_data["username"]
        if User.objects.filter(username=username).exists():
            raise ValidationError(f"입력한 아이디 ({username})는 이미 사용 중입니다")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"]
        if User.objects.filter(email=email).exists():
            raise ValidationError(f"입력한 이메일 ({email})은 이미 사용 중입니다")
        return email

    def clean(self):
        password1 = self.cleaned_data["password1"]
        password2 = self.cleaned_data["password2"]
        if password1 != password2:
            self.add_error("password2", "비밀번호와 비밀번호 확인란의 값이 다릅니다")

    def save(self):
        username = self.cleaned_data["username"]
        email = self.cleaned_data["email"]
        password1 = self.cleaned_data["password1"]
        phone_number = self.cleaned_data["phone_number"]
        profile_image = self.cleaned_data["profile_image"]
        short_description = self.cleaned_data["short_description"]
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            phone_number=phone_number,
            profile_image=profile_image,
            short_description=short_description,
        )
        return user
