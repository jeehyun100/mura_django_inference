from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from uploads.core import views
from . import shared_values as shared


urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    #url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    url(r'^uploads/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from .model_load.mura_model import MuraModel

model = MuraModel()
shared.model_arr = [model.load()]
print("model loading")
# if model_cnt > 0:
#     from model_load.mura_model import MuraModel
#     preload_model = FaceModel(threshold=1.24, model_arr=model_arr, licence=licence)
#     init_value.model_arr = model_arr