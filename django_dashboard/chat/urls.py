from django.urls import path
from . import views

app_name = "chat"

urlpatterns = [
    path("", views.stats_dashboard, name="stats_dashboard"),
    path("chat/", views.index, name="index"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/rename/", views.rename_conversation, name="rename_conversation"),
    path("history/<int:pk>/delete/", views.delete_conversation, name="delete_conversation"),
    path("history/<int:pk>/export/", views.export_conversation, name="export_conversation"),
    path("source/", views.download_source, name="download_source"),
    path("corpus/", views.corpus, name="corpus"),
    path("corpus/upload/", views.upload_corpus, name="upload_corpus"),
    path("api/new/", views.new_conversation, name="new_conversation"),
    path("api/stream/", views.stream_chat, name="stream_chat"),
]