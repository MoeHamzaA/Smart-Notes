from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('ask_ai/', views.ask_ai, name='ask_ai'),
    path('process/', views.process_files, name='process_files'),
    path('flash_cards/', views.flash_cards, name='flash_cards'),
]
