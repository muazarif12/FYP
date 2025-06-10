# import time
# import whisper 
# import torch 
# import asyncio
# from constants import OUTPUT_DIR
# from summarizer import generate_response_async
# from utils import format_timestamp
# import os
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound




# async def get_youtube_transcript(video_id):
#     """Try to get transcript directly from YouTube if available and format it like Whisper output."""
#     try:
#         print("Attempting to fetch official YouTube transcript...")
#         transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.list_transcripts, video_id)
#         try:
#             transcript = await asyncio.to_thread(transcript_list.find_manually_created_transcript)
#         except:
#             try:
#                 transcript = await asyncio.to_thread(transcript_list.find_generated_transcript)
#             except:
#                 transcript = await asyncio.to_thread(transcript_list.find_transcript, ['en'])

#         transcript_data = await asyncio.to_thread(transcript.fetch)
#         transcript_language = transcript.language_code
#         print(f"Found YouTube transcript in language: {transcript_language}")

#         full_text = ""
#         full_transcript_with_timestamps = ""
#         whisper_style_segments = []

#         for item in transcript_data:
#             start = item.start
#             duration = getattr(item, 'duration', 0)
#             end = start + duration
#             text = item.text.replace("\n", " ")
#             full_text += text + " "
#             start_fmt = format_timestamp(start)
#             end_fmt   = format_timestamp(end)
#             full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}:  {text}\n\n"
#             whisper_style_segments.append((start, end, text))

#         # Save raw and timestamped transcripts
#         raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
#         with open(raw_path, "w", encoding="utf-8") as f:
#             f.write(full_text)
#         print(f"Full transcript saved to {raw_path}")

#         ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
#         with open(ts_path, "w", encoding="utf-8") as f:
#             f.write(full_transcript_with_timestamps)
#         print(f"Timestamped transcript saved to {ts_path}")

#         # Return: segments, timestamped, language, raw_text
#         return whisper_style_segments, full_transcript_with_timestamps, transcript_language, full_text

#     except (TranscriptsDisabled, NoTranscriptFound) as e:
#         print(f"No YouTube transcript available: {e}")
#         return None, None, None, None
#     except Exception as e:
#         print(f"Error getting YouTube transcript: {e}")
#         return None, None, None, None



# async def transcribe_video(video_path):
#     """Transcribe video using Whisper, return segments, timestamped transcript, language, and raw text."""
#     model = None
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"CUDA available: {torch.cuda.is_available()}")
#         model = whisper.load_model("large-v3-turbo").to(device)
#         print("Using device for Whisper:", device)
#         print(video_path)
#         start_time = time.time()
        
#         if not os.path.exists(video_path):
#             print(f"Error: Video file not found at {video_path}")
#             return None, None, None, None
            
#         print(f"Transcribing video from {video_path}...")
#         result = await asyncio.to_thread(
#             model.transcribe,
#             video_path,
#             language=None,
#             word_timestamps=True
#         )
        
#         detected_language = result['language']
#         print(f"Detected language: {detected_language}")
        
#         # Build timestamped transcript
#         full_transcript_with_timestamps = ""
#         segments = []
#         for seg in result['segments']:
#             start_fmt = format_timestamp(seg['start'])
#             end_fmt = format_timestamp(seg['end'])
#             text = seg['text']
#             full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}: {text}\n\n"
#             segments.append((seg['start'], seg['end'], text))
            
#         raw_text = result["text"]
        
#         # Save raw text
#         raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
#         with open(raw_path, "w", encoding="utf-8") as f:
#             f.write(raw_text)
#         print(f"Full transcript saved to {raw_path}")
        
#         # Save timestamped transcript
#         ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
#         with open(ts_path, "w", encoding="utf-8") as f:
#             f.write(full_transcript_with_timestamps)
#         print(f"Timestamped transcript saved to {ts_path}")
        
#         elapsed_time = time.time() - start_time
#         print(f"Transcription completed in {elapsed_time:.2f} seconds")
        
#         return segments, full_transcript_with_timestamps, detected_language, raw_text
        
#     finally:
#         # Properly clean up the model and free memory
#         if model is not None:
#             model.cpu()  # Move model back to CPU if it was on GPU
#             del model    # Delete the model reference
            
#             # Clear CUDA cache if GPU was used
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 print("Model unloaded and CUDA cache cleared")

import time
import torch 
import asyncio
from constants import OUTPUT_DIR
from summarizer import generate_response_async
from utils import format_timestamp
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from faster_whisper import WhisperModel




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
    """Transcribe video using faster-whisper, return segments, timestamped transcript, language, and raw text."""
    model = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Use faster-whisper with optimized settings
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        print("Using device for faster-whisper:", device)
        print("Compute type:", compute_type)
        print(video_path)
        start_time = time.time()
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, None, None, None
            
        print(f"Transcribing video from {video_path}...")
        
        # faster-whisper returns segments generator and info object
        segments_generator, info = await asyncio.to_thread(
            model.transcribe,
            video_path,
            language=None,  # Auto-detect language
            word_timestamps=True,
            beam_size=5,  # Good balance of speed and accuracy
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,  # Voice activity detection for better accuracy
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        detected_language = info.language
        print(f"Detected language: {detected_language}")
        print(f"Language probability: {info.language_probability:.2f}")
        
        # Process segments generator into lists
        full_transcript_with_timestamps = ""
        segments = []
        full_text = ""
        
        for segment in segments_generator:
            start_fmt = format_timestamp(segment.start)
            end_fmt = format_timestamp(segment.end)
            text = segment.text
            full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}: {text}\n\n"
            segments.append((segment.start, segment.end, text))
            full_text += text + " "
        
        # Save raw text
        raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_text.strip())
        print(f"Full transcript saved to {raw_path}")
        
        # Save timestamped transcript
        ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(full_transcript_with_timestamps)
        print(f"Timestamped transcript saved to {ts_path}")
        
        elapsed_time = time.time() - start_time
        print(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        return segments, full_transcript_with_timestamps, detected_language, full_text.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, None, None, None
        
    finally:
        # Clean up the model and free memory
        if model is not None:
            del model
            
            # Clear CUDA cache if GPU was used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Model unloaded and CUDA cache cleared")


# Optional: For even faster processing, you can create a service class 
# to keep the model loaded between transcriptions
class FastTranscriptionService:
    def __init__(self, model_size="large-v3"):
        self.model = None
        self.model_size = model_size
        
    def load_model(self):
        """Load the model if not already loaded."""
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            print(f"Loading {self.model_size} model on {device} with {compute_type} precision...")
            self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
    
    async def transcribe(self, video_path):
        """Transcribe using the cached model."""
        self.load_model()
        
        try:
            print(f"Transcribing video from {video_path}...")
            start_time = time.time()
            
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return None, None, None, None
            
            segments_generator, info = await asyncio.to_thread(
                self.model.transcribe,
                video_path,
                language=None,
                word_timestamps=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            detected_language = info.language
            print(f"Detected language: {detected_language}")
            
            # Process segments
            full_transcript_with_timestamps = ""
            segments = []
            full_text = ""
            
            for segment in segments_generator:
                start_fmt = format_timestamp(segment.start)
                end_fmt = format_timestamp(segment.end)
                text = segment.text
                full_transcript_with_timestamps += f"{start_fmt} - {end_fmt}: {text}\n\n"
                segments.append((segment.start, segment.end, text))
                full_text += text + " "
            
            # Save files
            raw_path = os.path.join(OUTPUT_DIR, "full_transcript.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text.strip())
            print(f"Full transcript saved to {raw_path}")
            
            ts_path = os.path.join(OUTPUT_DIR, "timestamped_transcript.txt")
            with open(ts_path, "w", encoding="utf-8") as f:
                f.write(full_transcript_with_timestamps)
            print(f"Timestamped transcript saved to {ts_path}")
            
            elapsed_time = time.time() - start_time
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            return segments, full_transcript_with_timestamps, detected_language, full_text.strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None, None, None, None
    
    def cleanup(self):
        """Clean up the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Model unloaded and CUDA cache cleared")


# Usage example for the service class:
# transcription_service = FastTranscriptionService("large-v3")
# result = await transcription_service.transcribe(video_path)
# # Keep service alive for multiple transcriptions...
# transcription_service.cleanup()  # Call when done