from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),      # GET  /     -> the HTML page
    path("ask", views.ask, name="ask"),     # POST /ask  -> JSON API
]
