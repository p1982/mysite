from django.urls import path
from . import views

app_name = 'face'
urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.result, name='results'),
    path('detect/', views.detect, name='detect'),
    path('classify/', views.classify_image, name='classify_image'),  # Route for classify.html
]
