# Generated by Django 3.0.9 on 2020-08-07 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0006_auto_20200807_1137'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='plot',
            name='preprocessed_data',
        ),
        migrations.RemoveField(
            model_name='plot',
            name='processed_data',
        ),
        migrations.AddField(
            model_name='plot',
            name='model_type',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
