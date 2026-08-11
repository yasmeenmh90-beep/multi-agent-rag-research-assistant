# Generated manually to match the project's existing migration style.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0004_message_response_time_s"),
    ]

    operations = [
        migrations.CreateModel(
            name="UploadRecord",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("domain", models.CharField(max_length=100)),
                ("files_saved", models.IntegerField(default=0)),
                ("chunks_added", models.IntegerField(default=0)),
                ("files_failed", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]

    