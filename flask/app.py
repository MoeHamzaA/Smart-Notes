from flask import Flask, request, jsonify, send_from_directory
import subprocess
import time
import threading
import os
from tqdm import tqdm
from pydub import AudioSegment
import speech_recognition as sr
from queue import Queue
from collections import OrderedDict

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



if not os.path.exists('uploads'):
    os.makedirs('uploads')

def allowed_file(filename):
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print(f"Checking if file is allowed: {result}")
    return result

def convert_with_ffmpeg(input_path, output_path):
    command = [
        'ffmpeg',
        '-v', 'debug',
        '-i', input_path,        # Input file
        '-ac', '1',              # Set audio to mono (1 channel)
        '-ar', '16000',          # Set audio sample rate to 16 kHz
        output_path              # Output file
    ]
    print(f"Running ffmpeg with command: {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"Conversion complete: {output_path}")

def measure_time_for_conversion(conversion_func, *args):
    start_time = time.time()
    conversion_func(*args)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken for conversion: {duration:.2f} seconds")
    return duration

def split_audio(wav_file_path, chunk_duration):
    print(f"Splitting audio file: {wav_file_path}")
    audio = AudioSegment.from_wav(wav_file_path)
    chunks = []
    num_chunks = len(audio) // (chunk_duration * 1000) + 1
    for i in range(num_chunks):
        chunk = audio[i * chunk_duration * 1000:(i + 1) * chunk_duration * 1000]
        chunk_path = wav_file_path.replace('.wav', f'_chunk{i + 1}.wav')
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        print(f"Created chunk {i + 1} ({chunk_duration // 60} minutes): {chunk_path}")
    return chunks

def transcribe_audio_chunk(wav_chunk_path, result_queue):
    print(f"Transcribing chunk: {wav_chunk_path}")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_chunk_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "[Unintelligible]"
        except sr.RequestError:
            transcript = "[Request Error]"
    chunk_id = int(wav_chunk_path.split('_chunk')[-1].split('.wav')[0])
    result_queue.put((chunk_id, transcript))
    print(f"Transcription for chunk {chunk_id}: {transcript}")

def transcribe_audio_in_chunks(wav_file_path, chunk_duration):
    chunks = split_audio(wav_file_path, chunk_duration)
    result_queue = Queue()

    def transcribe_chunk(chunk_path, progress_bar):
        transcribe_audio_chunk(chunk_path, result_queue)
        progress_bar.update(1)
        os.remove(chunk_path)  # Clean up the chunk file after processing

    # Create a progress bar
    print(f"Transcribing audio in chunks...")
    with tqdm(total=len(chunks), desc="Transcribing") as progress_bar:
        threads = []
        for chunk_path in chunks:
            thread = threading.Thread(target=transcribe_chunk, args=(chunk_path, progress_bar))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    # Collect results from the queue and store them in an ordered dictionary
    transcripts = OrderedDict()
    while not result_queue.empty():
        chunk_id, transcript = result_queue.get()
        transcripts[chunk_id] = transcript

    # Create a full transcript in the correct order
    full_transcript = ""
    for i in range(1, len(transcripts) + 1):
        full_transcript += f"Transcription for chunk {i}: {transcripts[i]}\n"
    print("Full transcription completed.")
    return full_transcript

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received file upload request.")
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        print("No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_file_path)
        print(f"File saved to {input_file_path}")

        output_ffmpeg_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output_ffmpeg.wav')

        # Measure conversion time with ffmpeg
        ffmpeg_time = measure_time_for_conversion(convert_with_ffmpeg, input_file_path, output_ffmpeg_file)

        # Transcribe audio using ffmpeg
        ffmpeg_transcript = transcribe_audio_in_chunks(output_ffmpeg_file, 300)

        return jsonify({'transcript': ffmpeg_transcript})

    print("Invalid file format.")
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
