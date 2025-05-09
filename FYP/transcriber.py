import time
import whisper 
import torch 
import asyncio
from constants import OUTPUT_DIR
from summarizer import generate_response_async
from utils import format_timestamp
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound




async def get_youtube_transcript(video_id):
    """Try to get transcript directly from YouTube if available and format it like Whisper output."""
    try:
        print("Attempting to fetch official YouTube transcript...")
        transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.list_transcripts, video_id)
        try:
            transcript = await asyncio.to_thread(transcript_list.find_manually_created_transcript)
        except:
            try:
                transcript = await asyncio.to_thread(transcript_list.find_generated_transcript)
            except:
                transcript = await asyncio.to_thread(transcript_list.find_transcript, ['en'])

        transcript_data = await asyncio.to_thread(transcript.fetch)
        transcript_language = transcript.language_code
        print(f"Found YouTube transcript in language: {transcript_language}")

        full_text = ""
        full_transcript_with_timestamps = ""
        whisper_style_segments = []

        for item in transcript_data:
            start = item.start
            duration = getattr(item, 'duration', 0)
            end = start + duration
            text = item.text.replace("\n", " ")
            full_text += text + " "
            start_fmt = format_timestamp(start)
            end_fmt   = format_timestamp(end)
            full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}:  {text}\n\n"
            whisper_style_segments.append((start, end, text))

        # Save raw and timestamped transcripts
        raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Full transcript saved to {raw_path}")

        ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(full_transcript_with_timestamps)
        print(f"Timestamped transcript saved to {ts_path}")

        # Return: segments, timestamped, language, raw_text
        return whisper_style_segments, full_transcript_with_timestamps, transcript_language, full_text

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No YouTube transcript available: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error getting YouTube transcript: {e}")
        return None, None, None, None



async def transcribe_video(video_path):
    """Transcribe video using Whisper, return segments, timestamped transcript, language, and raw text."""
    model = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        model = whisper.load_model("large-v3-turbo").to(device)
        print("Using device for Whisper:", device)
        print(video_path)
        start_time = time.time()
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, None, None, None
            
        print(f"Transcribing video from {video_path}...")
        result = await asyncio.to_thread(
            model.transcribe,
            video_path,
            language=None,
            word_timestamps=True
        )
        
        detected_language = result['language']
        print(f"Detected language: {detected_language}")
        
        # Build timestamped transcript
        full_transcript_with_timestamps = ""
        segments = []
        for seg in result['segments']:
            start_fmt = format_timestamp(seg['start'])
            end_fmt = format_timestamp(seg['end'])
            text = seg['text']
            full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}: {text}\n\n"
            segments.append((seg['start'], seg['end'], text))
            
        raw_text = result["text"]
        
        # Save raw text
        raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"Full transcript saved to {raw_path}")
        
        # Save timestamped transcript
        ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(full_transcript_with_timestamps)
        print(f"Timestamped transcript saved to {ts_path}")
        
        elapsed_time = time.time() - start_time
        print(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        return segments, full_transcript_with_timestamps, detected_language, raw_text
        
    finally:
        # Properly clean up the model and free memory
        if model is not None:
            model.cpu()  # Move model back to CPU if it was on GPU
            del model    # Delete the model reference
            
            # Clear CUDA cache if GPU was used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Model unloaded and CUDA cache cleared")