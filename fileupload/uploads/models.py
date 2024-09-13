# uploads/models.py
from django.db import models

class Upload(models.Model):
    doc_file = models.FileField(upload_to='documents/')
    video_file = models.FileField(upload_to='videos/')
    wav_file = models.FileField(upload_to='audio/', blank=True, null=True)  # Allow blank and null
    transcript_file = models.FileField(upload_to='transcripts/', blank=True, null=True)  # Add this line

    def __str__(self):
        return self.docx_pdf_file.name
