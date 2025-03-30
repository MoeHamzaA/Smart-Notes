# Smart-Notes

## Overview
A powerful study aid that lets you upload your notes and lecture videos to create more effective study materials. It cross-references your notes with video content to enhance their accuracy, generates helpful flashcards for better learning, and features a chatbox for personalized assistance with the Gemini AI.

This project is a web-based service that allows users to upload video files, convert them to audio, and transcribe the content. It uses a Flask backend to handle file uploads and process audio transcription in parallel chunks for improved performance.

## Features

- **Video to Audio Conversion**: Converts uploaded MP4 files to WAV format using FFmpeg
- **Audio Chunking**: Splits large audio files into manageable chunks for faster processing
- **Parallel Transcription**: Processes audio chunks simultaneously using multiple threads
- **Progress Tracking**: Visualizes transcription progress with a progress bar
- **REST API**: Simple API endpoint for file uploads and transcription requests

## Technical Components

### Dependencies

- Flask: Web framework for the API
- FFmpeg: Audio/video conversion tool
- pydub: Audio processing library
- SpeechRecognition: Library for transcribing audio to text
- tqdm: Progress bar visualization
- Threading: Parallel processing capability

### Processing Pipeline

1. **File Upload**: User uploads an MP4 file through the `/upload` endpoint
2. **Audio Extraction**: FFmpeg extracts and converts audio to a 16kHz mono WAV file
3. **Audio Splitting**: The WAV file is split into 5-minute chunks
4. **Parallel Transcription**: Each chunk is transcribed simultaneously using Google's speech recognition service
5. **Result Assembly**: Individual chunk transcriptions are assembled in the correct order
6. **Response**: The complete transcription is returned to the user as a JSON response

## API Endpoints

### POST `/upload`

Accepts MP4 video files and returns the transcription.

**Request:**
- Content-Type: multipart/form-data
- Body: file (MP4 video file)

**Response:**
```json
{
  "transcript": "Full transcription text..."
}
```

**Error Responses:**
```json
{
  "error": "No file part"
}
```
```json
{
  "error": "No selected file"
}
```
```json
{
  "error": "Invalid file format"
}
```

## Performance Considerations

- Audio files are split into 5-minute chunks for parallel processing
- The system measures and logs conversion time for performance monitoring
- Temporary chunk files are automatically deleted after processing to conserve storage

## Future Improvements

Potential enhancements for this service could include:

1. User authentication for secure access
2. Support for additional video/audio formats
3. Alternative transcription engines for better accuracy
4. Frontend interface for easier interaction
5. Persistent storage of transcriptions
6. Transcription editing capabilities
7. Export options for transcription results

## Setup and Deployment

To run this service locally:

1. Install the required dependencies:
   ```
   pip install flask pydub SpeechRecognition tqdm
   ```

2. Ensure FFmpeg is installed on your system

3. Run the Flask application:
   ```
   python app.py
   ```

4. The service will be available at `http://localhost:5000`

## Technical Notes

- The service currently uses Google's free speech recognition API, which has usage limitations
- Processing large files may take considerable time depending on server resources
- The system is designed for asynchronous processing to prevent timeouts
