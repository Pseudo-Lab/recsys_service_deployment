from django.db import models

class DaumMovies(models.Model):
    movieid = models.AutoField(db_column='movieId', primary_key=True)  # Field name made lowercase.
    titleko = models.CharField(db_column='titleKo', max_length=100, blank=True, null=True)  # Field name made lowercase.
    titleen = models.CharField(db_column='titleEn', max_length=100, blank=True, null=True)  # Field name made lowercase.
    synopsis = models.TextField(blank=True, null=True)
    cast = models.CharField(max_length=10000, blank=True, null=True)
    mainpageurl = models.TextField(db_column='mainPageUrl', blank=True, null=True)  # Field name made lowercase.
    posterurl = models.TextField(db_column='posterUrl', blank=True, null=True)  # Field name made lowercase.
    numofsiteratings = models.IntegerField(db_column='numOfSiteRatings', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'daum_movies'


# class WatchedMovie(models.Model):
#     name = models.CharField(max_length=30)
#
#
# class PopMovies(models.Model):
#     movieId = models.BigAutoField(primary_key=True)
#     mean = models.FloatField(null=True, blank=True)
#     count = models.BigIntegerField(null=True, blank=True)
