from django.urls import path,include

from . import views
from .views import Preprocessing,Showplot,ChooseModel,cards

app_name= 'prediction'


urlpatterns = [
    path('', Preprocessing.as_view(), name='preprocessing'),
    path('showplot/<type>',views.Showplot.as_view(),name='showplot'),
    path('choose_model',ChooseModel.as_view(),name="choose-model"),
    path('api-auth/', include('rest_framework.urls')),
    path('cards/',cards,name="cards")
]