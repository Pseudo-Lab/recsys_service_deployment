from django.shortcuts import render

from paper_review.models import Post
import markdown


def index(request):
    posts = Post.objects.all().order_by('-pk')
    return render(request=request,
                  template_name='post_list.html',
                  context={
                      'posts': posts
                  })


def single_post_page(request, pk):
    post = Post.objects.get(pk=pk)

    md_mapper = {
        1: 'post_markdowns/kprn review.md',
        2: 'post_markdowns/ngcf review.md',
        3: 'post_markdowns/sasrec review.md',
        4: 'post_markdowns/srgnn review.md',
        5: 'post_markdowns/bert4rec review.md'
    }
    md_file_path = md_mapper[pk]
    post.set_content_from_md_file(md_file_path)
    print(f"{type(post)}")
    return render(
        request=request,
        template_name='post_detail.html',
        context={
            'post': post
        }
    )
