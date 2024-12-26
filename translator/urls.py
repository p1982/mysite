from django.urls import path
from . import views


urlpatterns = [
    path('', views.translate_text, name='index'),
    path('translate/', views.translate_text, name='translate_text'),
]
