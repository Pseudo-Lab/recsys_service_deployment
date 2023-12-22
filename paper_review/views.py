from django.shortcuts import render

from paper_review.models import Post

def index(request):
    posts = Post.objects.all().order_by('-pk')
    return render(request=request,
                  template_name='paper_review/post_list.html',
                  context={
                      'posts': posts
                  })

def single_post_page(request, pk):
    post = Post.objects.get(pk=pk)

    return render(
        request=request,
        template_name='paper_review/post_detail.html',
        context={
            'post': post
        }
    )
