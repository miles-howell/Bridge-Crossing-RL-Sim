# simulation/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/state/', views.api_state, name='api_state'),
    path('api/q_values/', views.api_q_values, name='api_q_values'),
]