# Generated by Django 5.0 on 2024-07-31 05:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uploads', '0003_alter_upload_wav_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='upload',
            name='transcript_file',
            field=models.FileField(blank=True, null=True, upload_to='transcripts/'),
        ),
    ]
