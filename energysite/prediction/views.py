from django.shortcuts import render, redirect
from pandas import read_csv
from scripts.model_processing import split_dataset, summarize_scores
from scripts.model_processing import evaluate_model as ev_model
from scripts.prepare_dataset import problem_frame
from scripts.encoder_decoder_model import evaluate_model as encode_decoder
from matplotlib import pyplot
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.conf.urls.static import static
from django.conf import settings as djangoSettings
from django.contrib import messages
from .models import Plot, LstmModels
from django.views import View
import io
import base64
import urllib
# Create your views here.


def index(request):

    return render(request, "home.html")


def cards(request):
    return render(request, "model_cards1.html")


class Preprocessing(View):

    def get(self, request, *args, **kwargs):
        return render(request, "preprocessing.html")

    def post(self, request, *args, **kwargs):

        csv = request.FILES['fileToUpload']
        dataset = read_csv(csv,
                           sep=';', header=0, low_memory=False,
                           infer_datetime_format=True,
                           parse_dates={'datetime': [0, 1]},
                           index_col=['datetime'])

        csv = problem_frame(dataset)
        plot = Plot()
        plot.csv = csv
        plot.save()
        return redirect("prediction:choose-model")


class ChooseModel(View):
    def get(self, request, *args, **kwargs):
        try:
            models = LstmModels.objects.all()
            return render(request, "model_cards.html", {'models': models})
        except Exception as e:
            return redirect("prediction:preprocessing")


class Showplot(View):
    def get(self, request, *args, **kwargs):

        if kwargs['type']:
            plot = Plot.objects.order_by('-date_created')[0]
            csv = plot.csv
            model_type = kwargs['type']
            dataset = read_csv(
                r'S:\Practica\EnergyPrediction\energyprediction\energysite\household_power_consumption_days.csv',
                header=0,
                infer_datetime_format=True,
                parse_dates=['datetime'],
                index_col=['datetime'])

            values = dataset.values

            train, test = split_dataset(values)
            n_input = 14
            if model_type == "lstm_encoder_decoder":
                score_rmse, scores_rmse, score_mae, scores_mae = encode_decoder(
                    train, test, n_input)
            else:
                score_rmse, scores_rmse, score_mae, scores_mae = ev_model(
                    train, test, n_input)

            # Plot the scores
            days = ['sun', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat']

            pyplot.plot(days, scores_mae, marker='o', label='lstm')
            # pyplot.plot(range(10))
            fig = pyplot.gcf()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)

            pyplot.plot(days, scores_rmse, marker='o', label='lstm')
            # pyplot.plot(range(10))
            fig = pyplot.gcf()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)

            plot.uri_rmse = uri

            plot.save()
            all_entries = Plot.objects.all()
            return render(
                request, 'show_prediction.html', {
                    'data': Plot.objects.all()})
        else:

            messages.warning(
                request, "The uploaded file has not the right format")
            return render(request, "home.html")
