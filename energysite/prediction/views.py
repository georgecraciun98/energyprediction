from django.shortcuts import render,redirect
from pandas import read_csv
from scripts.model_processing import split_dataset,evaluate_model,summarize_scores
from scripts.prepare_dataset import problem_frame
from matplotlib import pyplot
from django.conf.urls.static import static
from django.contrib import messages
from .models import Plot
from django.views import View
import io
import base64
import urllib
# Create your views here.

def index(request):
    
    return render(request,"home.html")

class Preprocessing(View):

    def get(self,request,*args,**kwargs):
        return render(request,"preprocessing.html")

    def post(self,request,*args,**kwargs):
        
        
        csv = request.FILES['fileToUpload']
        dataset = read_csv(csv,
                   sep=';', header=0, low_memory=False, 
                   infer_datetime_format=True, 
                   parse_dates={'datetime':[0,1]}, 
                   index_col=['datetime'])

        csv=problem_frame(dataset)
        return redirect("prediction:showplot",csv=csv)

class Showplot(View):
    def get(self,request,*args,**kwargs):

        try:
            if kwargs['csv']:
                csv=kwargs['csv']
                dataset=read_csv(csv,header=0,
                        infer_datetime_format=True
                        ,parse_dates=['datetime'],index_col=['datetime'])

                values=dataset.values

                train,test=split_dataset(values)


                n_input=14
                score,scores=evaluate_model(train,test,n_input)

                #Plot the scores
                days=['sun','mon','tue','wed','thur','fri','sat']
                summarize_scores('lstm', score, scores)

                pyplot.plot(days,scores,marker='o',label='lstm')
                # pyplot.plot(range(10))
                fig=pyplot.gcf()

                buf=io.BytesIO()
                fig.savefig(buf,format="png")
                buf.seek(0)
                string=base64.b64encode(buf.read())
                plot=Plot()
                uri=urllib.parse.quote(string)
                plot.uri=uri
                
                plot.save()
                all_entries=Plot.objects.all()
                return render(request, 'show_prediction.html',{'data':Plot.objects.all()})
            else:

                messages.warning(request,"The uploaded file has not the right format")
                return render (request,"home.html")
        except:
            messages.warning(request,"We've encountered an error during processing")
            return render (request,"home.html")

        
    


