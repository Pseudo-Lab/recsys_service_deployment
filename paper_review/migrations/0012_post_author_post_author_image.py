# Generated by Django 5.1 on 2024-10-28 08:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('paper_review', '0011_comment'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='author',
            field=models.CharField(default='작성자추가', max_length=50),
        ),
        migrations.AddField(
            model_name='post',
            name='author_image',
            field=models.ImageField(blank=True, upload_to='paper_review/author_imgs', verbose_name='작성자 이미지'),
        ),
    ]