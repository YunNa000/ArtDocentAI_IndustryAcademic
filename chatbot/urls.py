from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home',views.home, name='home'),
    path('favorite',views.favorite, name='favorite'),
    path('IntroTeam',views.IntroTeam, name='IntroTeam'),
    path('IntroArtist',views.IntroArtist, name='IntroArtist'),
    path('chattrain', views.chattrain, name='chattrain'),
    path('chatanswer', views.chatanswer, name='chatanswer'),
]