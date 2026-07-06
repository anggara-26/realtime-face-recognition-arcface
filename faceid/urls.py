from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/enroll/", views.enroll, name="enroll"),
    path("api/recognize/", views.recognize, name="recognize"),
    path("api/persons/", views.list_persons, name="list_persons"),
    path("api/persons/<int:person_id>/", views.delete_person, name="delete_person"),
]
