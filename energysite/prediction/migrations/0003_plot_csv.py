# Generated by Django 3.0.9 on 2020-08-05 08:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0002_auto_20200804_1223'),
    ]

    operations = [
        migrations.AddField(
            model_name='plot',
            name='csv',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
    ]
