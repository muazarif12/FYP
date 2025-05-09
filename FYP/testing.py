import whisper
import torch
import yt_dlp
import os
import time
from vosk import Model, KaldiRecognizer
import wave
import json

# --- YouTube Downloader ---
def download_youtube_video(url, download_path="downloads"):
    """Download YouTube video/audio and return file path"""
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(download_path, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(result).replace('.webm', '.wav').replace('.mp4', '.wav')
    
    print(f"Downloaded audio: {audio_file}")
    return audio_file

# --- Whisper-only Transcription ---
def transcribe_with_whisper(video_path):
    """Pure Whisper transcription with timing"""
    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v3-turbo").to(device)
    
    print("Starting Whisper transcription...")
    result = model.transcribe(video_path)
    
    elapsed = time.time() - start_time
    print(f"Whisper completed in {elapsed:.2f} seconds")
    
    return result['text'], result['language'], elapsed

# --- Hybrid Transcription ---
def transcribe_hybrid(video_path, vosk_model_dir="FYP"):
    """Vosk (fast draft) + Whisper (accurate refinement)"""
    # Convert to 16kHz WAV for Vosk
    audio_16k = "temp_16k.wav"
    os.system(f"ffmpeg -i {video_path} -ar 16000 -ac 1 {audio_16k} -y")
    
    # --- Phase 1: Vosk Draft ---
    vosk_start = time.time()
    model_path = os.path.join(vosk_model_dir, "vosk-model-en-us-0.22")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}")
    
    model = Model(model_path)
    wf = wave.open(audio_16k, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    draft = []
    while True:
        data = wf.readframes(4000)
        if not data: break
        if rec.AcceptWaveform(data):
            draft.append(json.loads(rec.Result())["text"])
    
    vosk_elapsed = time.time() - vosk_start
    draft_text = " ".join(draft)
    print(f"Vosk draft generated in {vosk_elapsed:.2f}s")
    
    # --- Phase 2: Whisper Refinement ---
    whisper_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base").to(device)
    
    result = model.transcribe(video_path, initial_prompt=draft_text[:200])
    
    hybrid_elapsed = time.time() - whisper_start
    total_time = time.time() - vosk_start
    print(f"Whisper refinement completed in {hybrid_elapsed:.2f}s")
    print(f"Total hybrid time: {total_time:.2f}s")
    
    os.remove(audio_16k)
    return result['text'], result['language'], total_time

# --- Main Program ---
if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    
    # Download audio
    audio_path = download_youtube_video(youtube_url)
    
    # Choose mode
    print("\n1. Whisper-only (accurate but slower)")
    print("2. Hybrid (Vosk draft + Whisper refinement)")
    choice = input("Select mode (1/2): ")
    
    if choice == "1":
        transcript, lang, elapsed = transcribe_with_whisper(audio_path)
        mode = "Whisper-only"
    else:
        transcript, lang, elapsed = transcribe_hybrid(audio_path)
        mode = "Hybrid"
    
    # Save results
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"Processing Time: {elapsed:.2f}s\n")
        f.write(f"Detected Language: {lang}\n\n")
        f.write(transcript)
    
    print(f"\nTranscript saved to 'transcript.txt'")
    print(f"Total processing time: {elapsed:.2f} seconds")