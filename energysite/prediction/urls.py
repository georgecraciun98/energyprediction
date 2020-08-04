from django.urls import path

from . import views
from .views import Preprocessing,Showplot

app_name= 'prediction'


urlpatterns = [
    path('', Preprocessing.as_view(), name='preprocessing'),
    path('showplot/<csv>',views.Showplot.as_view(),name='showplot')
]