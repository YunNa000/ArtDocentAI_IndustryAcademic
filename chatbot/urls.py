from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chattrain', views.chattrain, name='chattrain'),
    path('chatanswer', views.chatanswer, name='chatanswer'),
]