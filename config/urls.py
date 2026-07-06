from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import HttpResponse
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "robots.txt",
        lambda request: HttpResponse("User-agent: *\nDisallow: /\n", content_type="text/plain"),
    ),
    path("", include("faceid.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
