from django.db import models

# Create your models here.


class Plot(models.Model):

    csv = models.FileField(blank=True, null=True)
    input_days = models.IntegerField(blank=True, null=True, default=14)
    preprocessed_data = models.FileField(blank=True, null=True)
    framed_data = models.FileField(blank=True, null=True)
    processed_data = models.FileField(blank=True, null=True)
    uri_rmse = models.CharField(max_length=250, blank=True, null=True)
    uri_mae = models.CharField(max_length=250, blank=True, null=True)
    date_created = models.DateField(auto_now_add=True)


class LstmModels(models.Model):
    model_type = models.CharField(max_length=20)
    description = models.CharField(max_length=300)
