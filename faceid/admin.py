from django.contrib import admin

from .models import Person


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("name", "engine", "num_source_photos", "det_score", "created_at")
    list_filter = ("engine",)
    search_fields = ("name",)
