from django.db import models

# Create your models here.

class Plot(models.Model):

    preprocessed_data=models.FileField(blank=True,null=True)
    framed_data=models.FileField(blank=True,null=True)
    processed_data=models.FileField(blank=True,null=True)
    uri=models.CharField(max_length=250)
    date_created=models.DateField(auto_now_add=True)