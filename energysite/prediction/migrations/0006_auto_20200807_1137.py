# Generated by Django 3.0.9 on 2020-08-07 08:37

from django.db import migrations, models
import scripts.storage


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0005_auto_20200806_1246'),
    ]

    operations = [
        migrations.AlterField(
            model_name='plot',
            name='csv',
            field=models.FileField(blank=True, null=True, storage=scripts.storage.OverWriteStorage(), upload_to=''),
        ),
        migrations.AlterField(
            model_name='plot',
            name='framed_data',
            field=models.FileField(blank=True, null=True, storage=scripts.storage.OverWriteStorage(), upload_to=''),
        ),
    ]