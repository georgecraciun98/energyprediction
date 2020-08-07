# -*- coding: utf-8 -*-

from pandas import read_csv
import numpy as np
from numpy import nan
from numpy import isnan
import os
from django.conf import settings
from prediction.models import Plot


def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            current = values[row][col]
            if isnan(current):
                current = values[row - one_day][col]


def prepare_data(dataset):

    dataset.replace(to_replace='?', value=nan, inplace=True)

    dataset = dataset.astype('float32')

    # Replace a missing value with a value from one day ago

    fill_missing(dataset.values)

    values = dataset.values
    dataset['sub_metering_4'] = (
        values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])
    # save updated dataset

    return dataset


def problem_frame(dataset):
    dataset = prepare_data(dataset)
    csv_path=os.path.join(settings.MEDIA_ROOT,'household_power_consumption.csv')
    dataset.to_csv(csv_path)
    plot=Plot.objects.order_by('-date_created')[0]
    
    dataset = read_csv(
        csv_path,
        header=0,
        infer_datetime_format=True,
        parse_dates=['datetime'],
        index_col=['datetime'])

    daily_groups = dataset.resample('D')
    daily_data = daily_groups.sum()

    daily_data.to_csv(csv_path)
    plot.framed_data=csv_path
    plot.save()

