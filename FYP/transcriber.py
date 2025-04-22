import time
import whisper, torch, asyncio
from constants import OUTPUT_DIR
from summarizer import generate_response_async
from utils import format_timestamp
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v3-turbo").to(device)


async def get_youtube_transcript(video_id):
    """Try to get transcript directly from YouTube if available and format it like Whisper output."""
    try:
        print("Attempting to fetch official YouTube transcript...")
        # Get available transcripts
        transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.list_transcripts, video_id)
        
        # First try to get a manual transcript in any language (usually higher quality)
        try:
            transcript = await asyncio.to_thread(transcript_list.find_manually_created_transcript)
        except:
            # If no manual transcript, try to get an auto-generated one
            try:
                transcript = await asyncio.to_thread(transcript_list.find_generated_transcript)
            except:
                # If no specific transcript found, just get the default one
                transcript = await asyncio.to_thread(transcript_list.find_transcript, ['en'])
        
        # Get the actual transcript data - this returns a list of dicts
        transcript_data = await asyncio.to_thread(transcript.fetch)
        
        # Get the language of the transcript
        transcript_language = transcript.language_code
        print(f"Found YouTube transcript in language: {transcript_language}")
        
        # Format transcript segments similar to Whisper output
        full_text = ""
        full_transcript_with_timestamps = ""
        
        # Create whisper-like segments structure
        whisper_style_segments = []
        
        for item in transcript_data:
            start = item.start
            duration = getattr(item, 'duration', 0)
            end = start + duration
            
            # YouTube often has line breaks in the text - replace them with spaces
            # This ensures each segment is on a single line like Whisper
            text = item.text.replace("\n", " ")
            
            # Add to full text
            full_text += text + " "
            
            # Format for timestamped transcript to match Whisper format exactly
            start_formatted = format_timestamp(start)
            end_formatted = format_timestamp(end)
            
            # Add the current segment with proper formatting - exactly two spaces after the colon
            # and a blank line between entries to match Whisper format
            full_transcript_with_timestamps += f"{start_formatted} - {end_formatted}:  {text}\n\n"
            
            # Create Whisper-like segment
            whisper_style_segments.append((start, end, text))
        
        # Save raw text
        raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Full transcript saved to {raw_path}")
        
        # Save timestamped transcript
        ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(full_transcript_with_timestamps)
        print(f"Timestamped transcript saved to {ts_path}")
        
        return whisper_style_segments, full_transcript_with_timestamps, transcript_language
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No YouTube transcript available: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error getting YouTube transcript: {e}")
        return None, None, None
    
# Transcribe video using Whisper with parallel processing
async def transcribe_video(video_path):
    print("Using device for Whisper:", device)
    start_time = time.time()  # Start time for transcription

    # Automatically detect language and transcribe with parallelism
    result = await asyncio.to_thread(model.transcribe, video_path, language=None, word_timestamps=True)  # Use threading for heavy operations
    detected_language = result['language']  # Detect the language from Whisper's output
    print(f"Detected language: {detected_language}")

    # If transcription is not in English, pass it to LLM for grammar correction
    if detected_language != "en":
        print(f"Transcription in {detected_language} detected. Passing it to LLM for grammar correction.")
        result["text"] = await generate_response_async(f"Please fix the following transcription to improve its overall quality. Correct any grammar mistakes, enhance sentence structure, and improve clarity while ensuring the meaning remains unchanged. Make the text sound more natural and fluent, fixing any awkward phrasing or unclear parts. If there are any spelling errors, please correct them as well. Below is the transcription in {detected_language}: {result['text']}")

    end_time = time.time()  # End time for transcription
    print(f"Time taken for transcription: {end_time - start_time:.4f} seconds")

    # Create full transcript with formatted timestamps
    full_transcript_with_timestamps = ""

    for segment in result['segments']:
        start_formatted = format_timestamp(segment['start'])
        end_formatted = format_timestamp(segment['end'])
        full_transcript_with_timestamps += f"{start_formatted} - {end_formatted}: {segment['text']}\n\n"

    # save raw text
    raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Full transcript saved to {raw_path}")

    # save timestamped transcript
    ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
    with open(ts_path, "w", encoding="utf-8") as f:
        f.write(full_transcript_with_timestamps)
    print(f"Timestamped transcript saved to {ts_path}")
    
    # Return timestamped segments with formatted timestamps and the detected language
    segments = [(segment['start'], segment['end'], segment['text']) for segment in result['segments']]
    return segments, full_transcript_with_timestamps, detected_language