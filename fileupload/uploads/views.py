from collections import Counter
import io
import os
import shutil
import tempfile
from django.shortcuts import render
from django.http import HttpResponse
from .models import Upload
from .forms import UploadForm
import fitz  # PyMuPDF
import docx
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import spacy
import moviepy.editor as mp
import google.generativeai as genai
from django.core.files import File
from django.core.files.storage import default_storage
from django.conf import settings
import numpy as np
import logging
import boto3
import time
import tqdm
import random
import subprocess
import requests

logger = logging.getLogger(__name__)

# Initialize models and pipelines
summarizer = pipeline("summarization")
nlp = spacy.load("en_core_web_sm")
genai.configure(api_key="AIzaSyCzpFkJ4gqT0W7TQRFP18twEcBXNgHXovA")
model = genai.GenerativeModel('gemini-1.5-flash')
# URL of the Flask endpoint that handles MP4 file processing
FLASK_SERVER_URL = 'http://127.0.0.1:5000/upload'  # Update this to your Flask server URL

def get_s3_client():
    # Create and return the S3 client
    return boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME
    )

def delete_folder_from_s3(bucket_name, folder_prefix):
    s3 = get_s3_client()
    try:
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
        if 'Contents' in result:
            keys = [obj['Key'] for obj in result['Contents']]
            if keys:
                s3.delete_objects(Bucket=bucket_name, Delete={'Objects': [{'Key': key} for key in keys]})
                logger.info(f"Deleted {len(keys)} items from '{folder_prefix}'")
        else:
            logger.info(f"No items found in '{folder_prefix}'")
    except Exception as e:
        logger.error(f"Error deleting folder from S3: {str(e)}")

def delete_local_folder(folder_path):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)
        logger.info(f"Deleted local folder and its contents: {folder_path}")
    else:
        logger.info(f"Local folder not found: {folder_path}")

def list_folders_in_s3():
    s3 = get_s3_client()
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    try:
        result = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        folders = [content['Prefix'] for content in result.get('CommonPrefixes', [])]
        return folders
    except Exception as e:
        logger.error(f"Error listing folders in S3: {str(e)}")
        return []

def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def transcribe_audio(wav_file):
    recognizer = sr.Recognizer()
    try:
        with io.BytesIO(wav_file.read()) as audio_file:
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
                logger.info("Successfully transcribed audio.")
                return transcript
    except sr.UnknownValueError:
        logger.error("Could not understand audio.")
        return "Could not understand audio"
    except sr.RequestError:
        logger.error("Could not request results.")
        return "Could not request results"
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return f"Error transcribing audio: {str(e)}"

def convert_mp4_to_wav(mp4_file, wav_file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_mp4_file:
        shutil.copyfileobj(mp4_file, temp_mp4_file)
        temp_mp4_file_path = temp_mp4_file.name

    temp_wav_file_path = os.path.splitext(temp_mp4_file_path)[0] + '.wav'

    try:
        command = [
            'ffmpeg',
            '-i', temp_mp4_file_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            temp_wav_file_path
        ]
        subprocess.run(command, check=True)

        if not os.path.exists(temp_wav_file_path):
            raise FileNotFoundError(f"The WAV file was not created: {temp_wav_file_path}")

        wav_file_dir = os.path.dirname(wav_file_name)
        if not os.path.exists(wav_file_dir):
            os.makedirs(wav_file_dir)

        shutil.move(temp_wav_file_path, wav_file_name)
        logger.info(f"Converted MP4 to WAV: {wav_file_name}")

        return temp_mp4_file_path, wav_file_name

    finally:
        if os.path.exists(temp_mp4_file_path):
            os.remove(temp_mp4_file_path)
        if os.path.exists(temp_wav_file_path):
            os.remove(temp_wav_file_path)

def compare_texts(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_keywords(text):
    # Use SpaCy to analyze the text
    doc = nlp(text)
    
    # Extract noun chunks and named entities
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    named_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
    keywords = noun_chunks + named_entities
    
    # Check if the text is empty or contains only stop words
    if not text.strip():
        logger.error("Text is empty or contains only stop words.")
        return set()

    # Initialize the vectorizer and fit_transform
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform([text])
    except ValueError as e:
        logger.error(f"Error with TfidfVectorizer: {str(e)}")
        return set()

    # Extract TF-IDF scores
    tfidf_scores = np.array(X.sum(axis=0)).flatten()
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))
    
    # Filter keywords based on TF-IDF scores
    tfidf_keywords = [word for word in keywords if tfidf_scores_dict.get(word, 0) > 0.1]
    all_keywords = noun_chunks + named_entities + tfidf_keywords
    keyword_freq = Counter(all_keywords)
    
    # Return keywords appearing more than once
    return set(keyword for keyword, freq in keyword_freq.items() if freq > 1)

def generate_suggestions(missing_keywords, docx_text, audio_text):
    missing_keywords_str = ', '.join(missing_keywords)
    prompt = (
        f"The following keywords are missing from the document: {missing_keywords_str}. "
        f"Here is the content of the document: {docx_text}. "
        f"Here is the content of the lecture transcript from the audio file: {audio_text}. "
        f"Please provide bullet point suggestions on what to add to the document to address the missing keywords. "
        f"Focus on integrating relevant information from the lecture transcript into the document to make it more complete."
    )
    response = model.generate_content(prompt)
    suggestions = response.text.split('\n')
    return [s.strip() for s in suggestions if s.strip()]

def is_file_in_use(file_path):
    try:
        with open(file_path, 'r+'):
            pass
    except IOError:
        return True
    return False

import os
import requests

def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload_instance = form.save(commit=False)
            upload_instance.save()

            if 'action' in request.POST:
                action = request.POST['action']
                
                audio_text = ""
                
                # Process video file if present
                if upload_instance.video_file:
                    mp4_file_name = upload_instance.video_file.name
                    wav_file_name = os.path.join('wav_files', os.path.splitext(mp4_file_name)[0] + '.wav')

                    # Send the MP4 file to the Flask server
                    with default_storage.open(mp4_file_name, 'rb') as mp4_file:
                        response = requests.post(FLASK_SERVER_URL, files={'file': mp4_file})

                    if response.status_code == 200:
                        audio_text = response.text
                    else:
                        print(f"Error: Failed to get transcript from Flask server. Status code: {response.status_code}")

                    # Save the transcript file
                    transcript_file_name = f"transcripts/{upload_instance.id}_transcript.txt"
                    s3 = get_s3_client()
                    s3.put_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=transcript_file_name, Body=audio_text)
                    
                docx_text = ""
                
                # Process document file if present
                if upload_instance.doc_file:
                    if upload_instance.doc_file.name.endswith('.pdf'):
                        docx_text = extract_text_from_pdf(upload_instance.doc_file)
                    elif upload_instance.doc_file.name.endswith('.docx'):
                        docx_text = extract_text_from_docx(upload_instance.doc_file)

                # Handle the action based on the button clicked
                if action == 'generate_flashcards':
                    # Process document file for flashcards if provided
                    if upload_instance.doc_file:
                        keywords = extract_keywords(docx_text)
                        flashcards = generate_flashcards(keywords)
                        definitions = generate_definitions(keywords)
                        request.session['definitions'] = definitions

                    # Process video file for flashcards if provided
                    if upload_instance.video_file:
                        audio_transcript = transcribe_audio(upload_instance.video_file)
                        keywords = extract_keywords(audio_transcript)
                        flashcards = generate_flashcards(keywords)
                        definitions = generate_definitions(keywords)
                        request.session['definitions'] = definitions

                    # Render flashcards page
                    return render(request, 'uploads/flash_cards.html', {
                        'flashcards_data': [{'term': key, 'definition': value} for key, value in definitions.items() if value.strip()],
                        'docx_keywords': list(extract_keywords(docx_text)) if upload_instance.doc_file else [],
                        'audio_keywords': list(extract_keywords(audio_transcript)) if upload_instance.video_file else []
                    })

                elif action == 'upload':
                    if upload_instance.video_file:
                        audio_text = transcribe_audio(upload_instance.video_file) if not audio_text else audio_text
                        docx_text = extract_text_from_pdf(upload_instance.doc_file) if upload_instance.doc_file and upload_instance.doc_file.name.endswith('.pdf') else ""
                        docx_text = extract_text_from_docx(upload_instance.doc_file) if upload_instance.doc_file and upload_instance.doc_file.name.endswith('.docx') else ""
                        similarity_score = compare_texts(docx_text, audio_text)
                        docx_keywords = extract_keywords(docx_text)
                        audio_keywords = extract_keywords(audio_text)
                        missing_keywords = audio_keywords - docx_keywords
                        suggestions = generate_suggestions(missing_keywords, docx_text, audio_text)

                        # Clean up S3 and local files
                        delete_folder_from_s3(settings.AWS_STORAGE_BUCKET_NAME, 'audio/')
                        delete_folder_from_s3(settings.AWS_STORAGE_BUCKET_NAME, 'fileupload/')
                        delete_folder_from_s3(settings.AWS_STORAGE_BUCKET_NAME, 'videos/')
                        delete_local_folder('media')

                        return render(request, 'uploads/success.html', {
                            'upload': upload_instance,
                            'similarity_score': similarity_score,
                            'missing_keywords': list(missing_keywords),
                            'suggestions': suggestions
                        })

    else:
        form = UploadForm()
    return render(request, 'uploads/upload.html', {'form': form})


def process_files(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload_instance = form.save(commit=False)

            mp4_file_path = None
            wav_file_path = None
            try:
                mp4_file = upload_instance.video_file
                wav_file_name = os.path.join('wav_files', os.path.splitext(mp4_file.name)[0] + '.wav')  # Ensure proper path
                wav_file = default_storage.open(wav_file_name, 'wb+')

                # Send the MP4 file to the Flask server
                response = requests.post(FLASK_SERVER_URL, files={'file': mp4_file})
                if response.status_code == 200:
                    audio_text = response.text
                else:
                    audio_text = "Error: Failed to get transcript from Flask server"

                # Process and save the results
                with default_storage.open(wav_file_name, 'wb+') as wav_file:
                    wav_file.write(response.content)

                transcript_file_name = f"transcripts/{upload_instance.id}_transcript.txt"
                with default_storage.open(transcript_file_name, 'w') as transcript_file:
                    transcript_file.write(audio_text)
                upload_instance.transcript_file = transcript_file_name
                upload_instance.save()

                docx_text = ""
                if upload_instance.doc_file.name.endswith('.pdf'):
                    docx_text = extract_text_from_pdf(upload_instance.doc_file)
                elif upload_instance.doc_file.name.endswith('.docx'):
                    docx_text = extract_text_from_docx(upload_instance.doc_file)

                similarity_score = compare_texts(docx_text, audio_text)
                docx_keywords = extract_keywords(docx_text)
                
                audio_keywords = extract_keywords(audio_text)
                missing_keywords = audio_keywords - docx_keywords
                suggestions = generate_suggestions(missing_keywords, docx_text, audio_text)

                # Delete the "audio/" and "videos/" folders from S3
                delete_folder_from_s3(settings.AWS_STORAGE_BUCKET_NAME, 'audio/')
                delete_folder_from_s3(settings.AWS_STORAGE_BUCKET_NAME, 'videos/')

                # Clean up local folders
                delete_local_folder('videos/')
                delete_local_folder('wav_files/')

                return render(request, 'uploads/success.html', {
                    'upload': upload_instance,
                    'similarity_score': similarity_score,
                    'missing_keywords': list(missing_keywords),
                    'suggestions': suggestions
                })

            finally:
                if wav_file_path and os.path.exists(wav_file_path) and not is_file_in_use(wav_file_path):
                    os.remove(wav_file_path)
                if mp4_file_path and os.path.exists(mp4_file_path) and not is_file_in_use(mp4_file_path):
                    os.remove(mp4_file_path)
                if upload_instance.video_file and not is_file_in_use(upload_instance.video_file.path):
                    default_storage.delete(upload_instance.video_file.name)
                if upload_instance.wav_file and not is_file_in_use(upload_instance.wav_file.path):
                    default_storage.delete(upload_instance.wav_file.name)

    else:
        form = UploadForm()
    return render(request, 'uploads/upload.html', {'form': form})

def upload_and_process_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)

        # Define Flask URL
        flask_url = 'http://localhost:5001/process'

        # Send file to Flask app
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(flask_url, files=files)

        if response.status_code == 200:
            # Handle the response (e.g., save the returned file)
            with open('received_transcript.txt', 'wb') as f:
                f.write(response.content)

            return HttpResponse("File processed and received.")
        else:
            return HttpResponse("Error processing file.")

    return render(request, 'upload.html')

def generate_flashcards(keywords):
    # Generate flashcards
    flashcards = []
    for keyword in keywords:
        definition = f"Definition for {keyword}"  # Replace with actual definition fetching
        flashcards.append({'term': keyword, 'definition': definition})
    return flashcards

def filter_keywords_for_flashcards(keywords):
    # Create a prompt for Gemini AI to filter helpful keywords
    prompt = (
        "Here is a list of keywords: {keywords}. "
        "Please filter out keywords that are not useful for creating flashcards. "
        "Focus on keywords that have general definitions or educational value. "
        "Return only the filtered keywords."
    ).format(keywords=', '.join(keywords))
    
    # Generate content using Gemini AI
    response = model.generate_content(prompt)
    filtered_keywords = response.text.split('\n')
    
    # Clean up the response and return filtered keywords
    filtered_keywords = [kw.strip() for kw in filtered_keywords if kw.strip()]
    return filtered_keywords

def generate_definitions(keywords):
    # Create a prompt for Gemini AI to generate definitions
    prompt = (
        "Here is a list of keywords: {keywords}. "
        "Please provide a brief, clear definition for each keyword. "
        "Return the definitions in the format: keyword: definition."
    ).format(keywords=', '.join(keywords))
    
    # Generate content using Gemini AI
    response = model.generate_content(prompt)
    definitions = response.text.strip().split('\n')
    
    # Parse the response and create a dictionary of definitions
    definitions_dict = {}
    for definition in definitions:
        if ':' in definition:
            keyword, def_text = definition.split(':', 1)
            definitions_dict[keyword.strip()] = def_text.strip()
    
    return definitions_dict


def ask_ai(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        if question:
            logger.info(f"Received question for AI: {question}")
            response = model.generate_content(question)
            logger.info(f"AI response generated.")
            return render(request, 'uploads/ask_ai.html', {'response': response.text})

    return render(request, 'uploads/ask_ai.html', {})

def flash_cards(request):
    return render(request, 'uploads/flash_cards.html')

def home(request):
    return render(request, 'uploads/home.html')