from django import forms


class UploadFileForm(forms.Form):
    file_csv=forms.FileField()


class SimpleForm(forms.Form):
    model_type=forms.CharField(max_length=20)