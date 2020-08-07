from django.db import models
from scripts.storage import OverWriteStorage
# Create your models here.


class Plot(models.Model):
    model_type= models.CharField(max_length=20,blank=True,null=True)
    csv = models.FileField(blank=True, null=True,storage=OverWriteStorage())
    input_days = models.IntegerField(blank=True, null=True, default=14)
    
    framed_data = models.FileField(blank=True, null=True,storage=OverWriteStorage())
    
    uri_rmse = models.CharField(max_length=250, blank=True, null=True)
    uri_mae = models.CharField(max_length=250, blank=True, null=True)
    date_created = models.DateField(auto_now_add=True)


class LstmModels(models.Model):
    model_type = models.CharField(max_length=20)
    description = models.CharField(max_length=300)
