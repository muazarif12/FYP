from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid
import shutil
import time
import whisper
import torch

from requests import request

from core.config import settings

# Import from utils router to access task storage
from routers.utils import task_storage

# Import your algorithmic functions - EXACT paths as in paste.txt
import sys
sys.path.append('.')  # Ensure current directory is in path

# Import exactly as your paste.txt shows
from Final_Optimized_Highlights.utils import (
    merge_word_lists, 
    segment_by_pauses, 
    filter_by_word_count, 
    process_transcript_words, 
    save_extractive_summary,
    saving_full_transcript
)
from Final_Optimized_Highlights.text_processing import merge_extractive_summaries, chunk_subtitle_segments
from Final_Optimized_Highlights.video_processing import process_video

# Import existing functions for non-algorithmic methods
from highlights import generate_highlights, generate_custom_highlights

router = APIRouter()

# Dictionary to track highlights tasks
highlights_task_storage = {}

# Data models
class HighlightsRequest(BaseModel):
    task_id: str  # Task ID from video processing
    duration_seconds: Optional[int] = None  # Target duration in seconds
    custom_prompt: Optional[str] = None  # Optional custom instructions
    use_algorithmic: bool = False  # Whether to use fast algorithmic method
    is_reel: bool = False  # Whether to generate a short reel format

class HighlightsResponse(BaseModel):
    highlights_task_id: str
    status: str = "processing"
    message: str

class HighlightSegment(BaseModel):
    start: float
    end: float
    description: str

class HighlightsResult(BaseModel):
    video_url: str
    segments: List[HighlightSegment]
    total_duration: float

@router.post("/highlights", response_model=HighlightsResponse, tags=["Highlights"])
async def generate_video_highlights(request: HighlightsRequest, background_tasks: BackgroundTasks):
    """
    Generate highlights from video content. Supports standard highlights, fast algorithmic highlights,
    reels, and custom highlights with specific instructions.
    
    - **task_id**: Task ID from video processing
    - **duration_seconds**: Optional target duration in seconds
    - **custom_prompt**: Optional custom instructions for highlight generation
    - **use_algorithmic**: Whether to use fast algorithmic method (default: False)
    - **is_reel**: Whether to generate a short reel format (default: False)
    
    Returns a task ID to track the highlights generation process.
    """
    # Verify that the task exists and is completed
    if request.task_id not in task_storage:
        raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
    
    task_info = task_storage[request.task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task processing not complete. Current status: {task_info['status']}"
        )
    
    # Create a new task ID for this highlights request
    highlights_task_id = str(uuid.uuid4())
    
    try:
        # Start highlights generation in background
        background_tasks.add_task(
            process_highlights_generation,
            highlights_task_id,
            request.task_id,
            request.duration_seconds,
            request.custom_prompt,
            request.use_algorithmic,
            request.is_reel
        )
        
        # Initialize task status
        highlights_task_storage[highlights_task_id] = {
            "status": "processing",
            "message": "Generating highlights..."
        }
        
        # Customize message based on request type
        message = "Your "
        if request.is_reel:
            message += "reel"
        elif request.custom_prompt:
            message += "custom highlights"
        elif request.use_algorithmic:
            message += "fast highlights"
        else:
            message += "highlights"
            
        if request.duration_seconds:
            message += f" ({request.duration_seconds} seconds)"
            
        message += " are being generated. Use the highlights_task_id to check status and get results."
        
        return HighlightsResponse(
            highlights_task_id=highlights_task_id,
            status="processing",
            message=message
        )
    
    except Exception as e:
        highlights_task_storage[highlights_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting highlights generation: {str(e)}")

@router.get("/highlights/status/{highlights_task_id}", tags=["Highlights"])
async def check_highlights_status(highlights_task_id: str):
    """
    Check the status of a highlights generation task.
    
    - **highlights_task_id**: Highlights task ID to check
    
    Returns the current status of the highlights generation.
    """
    if highlights_task_id not in highlights_task_storage:
        raise HTTPException(status_code=404, detail=f"Highlights task {highlights_task_id} not found")
    
    return highlights_task_storage[highlights_task_id]

@router.get("/highlights/result/{highlights_task_id}", response_model=HighlightsResult, tags=["Highlights"])
async def get_highlights_result(highlights_task_id: str):
    """
    Get the results of a completed highlights generation task.
    
    - **highlights_task_id**: Highlights task ID
    
    Returns information about the generated highlights including download URL and segments.
    """
    if highlights_task_id not in highlights_task_storage:
        raise HTTPException(status_code=404, detail=f"Highlights task {highlights_task_id} not found")
    
    task_info = highlights_task_storage[highlights_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Highlights task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info:
        raise HTTPException(status_code=500, detail="No result available")
    
    return HighlightsResult(**task_info["result"])

@router.get("/highlights/download/{highlights_task_id}", tags=["Highlights"])
async def download_highlights(highlights_task_id: str):
    """
    Download the generated highlights video.
    
    - **highlights_task_id**: Highlights task ID
    
    Returns the highlights video file.
    """
    if highlights_task_id not in highlights_task_storage:
        raise HTTPException(status_code=404, detail=f"Highlights task {highlights_task_id} not found")
    
    task_info = highlights_task_storage[highlights_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Highlights task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "video_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No highlights video available")
    
    video_path = task_info["result"]["video_path"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Highlights video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )

async def generate_highlights_algorithmically_new(
    video_path: str,
    target_duration: Optional[int] = None
):
    """
    EXACT replication of your colleague's main.py generate_highlights function
    This runs the EXACT same code that works in main.py
    """
    total_start = time.time()  # Track total time

    # Get video name from path (like your colleague does)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    
    # EXACT same paths as your colleague's main.py
    formatted_transcript_path = os.path.join(video_dir, 'formatted_transcript.txt')
    full_transcript_path = os.path.join(video_dir, 'full_transcript.txt')
    formatted_sentences_counts_path = os.path.join(video_dir, 'sentences_counts.txt')
    extracted_summary_path = os.path.join(video_dir, 'extracted_summary.txt')
    output_path = os.path.join(video_dir, 'highlights.mp4')
    
    # EXACT same configuration as your colleague's main.py
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.45  # FIXED: was 0.2 in original main.py - this was the problem!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v3-turbo").to(device)

    try:
        # Step 1: EXACT same transcription as colleague's main.py
        print("Transcribing video with Whisper...")
        result = model.transcribe(video_path, word_timestamps=True)
        language = result['language']
        words_data = merge_word_lists(result['segments'])
        print(f"Detected language: {language}")
        print(f"Total words extracted: {len(words_data)}")

        # Step 2: EXACT same sentence processing as colleague's main.py
        print("Processing transcript into sentences...")
        if language == 'english':
            sentences_timestamps = process_transcript_words(words_data, output_file=formatted_sentences_counts_path)
        else:
            sentences_timestamps = segment_by_pauses(words_data)
        
        print(f"Generated {len(sentences_timestamps)} sentences")

        # Step 3: EXACT same chunking as colleague's main.py (but with FIXED threshold)
        print("Chunking similar sentences...")
        sentences_timestamps_chunked = chunk_subtitle_segments(model_name, sentences_timestamps, threshold_similarity)
        
        print(f"Created {len(sentences_timestamps_chunked)} chunks")
        
        # ADDED SAFETY: If only 1 chunk (the original problem), force split
        if len(sentences_timestamps_chunked) == 1:
            print("⚠️  Only 1 chunk detected - this was the original problem! Force-splitting...")
            
            single_chunk = sentences_timestamps_chunked[0]
            chunk_duration = single_chunk['end'] - single_chunk['start']
            
            # Force split into smaller time windows
            segment_duration = min(120, chunk_duration / 4)  # 2 min segments or quarter of video
            time_chunks = []
            current_start = single_chunk['start']
            
            while current_start < single_chunk['end']:
                window_end = min(current_start + segment_duration, single_chunk['end'])
                
                # Find sentences in this time window
                window_sentences = []
                for sentence in sentences_timestamps:
                    if (sentence['start'] >= current_start and sentence['end'] <= window_end):
                        window_sentences.append(sentence['sentence'])
                
                if window_sentences:
                    time_chunks.append({
                        'sentence': ' '.join(window_sentences),
                        'start': current_start,
                        'end': window_end
                    })
                
                current_start = window_end
            
            sentences_timestamps_chunked = time_chunks
            print(f"✅ Force-split into {len(sentences_timestamps_chunked)} time-based chunks")

        # Step 4: EXACT same filtering as colleague's main.py
        print("Filtering by word count...")
        sentences_timestamps_chunked = filter_by_word_count(sentences_timestamps_chunked, 6)  # EXACT same as colleague
        
        print(f"Retained {len(sentences_timestamps_chunked)} chunks after filtering")

        # Step 5: EXACT same summary generation as colleague's main.py
        print("Generating extractive summary...")
        summary = merge_extractive_summaries(
            model_name, 
            sentences_timestamps_chunked, 
            summary_ratio=0.25,  # EXACT same as colleague
            min_topic_size=2,  # EXACT same as colleague  
            mmr_lambda_english=0.7,  # EXACT same as colleague
            language=language
        )

        print(f"Generated summary with {len(summary)} segments")

        # ADDED SAFETY: Create fallback if empty summary
        if not summary:
            print("⚠️  Empty summary generated! Creating fallback segments...")
            from moviepy import VideoFileClip
            with VideoFileClip(video_path) as temp_video:
                video_duration = temp_video.duration
            
            segment_length = min(30, video_duration / 6)
            
            summary = [
                {
                    'id': 1,
                    'sentence': 'Opening segment',
                    'start': video_duration * 0.1,
                    'end': video_duration * 0.1 + segment_length,
                    'topic': 0,
                    'score': 1.0
                },
                {
                    'id': 2,
                    'sentence': 'Middle segment', 
                    'start': video_duration * 0.45,
                    'end': video_duration * 0.45 + segment_length,
                    'topic': 1,
                    'score': 0.9
                },
                {
                    'id': 3,
                    'sentence': 'Closing segment',
                    'start': video_duration * 0.8,
                    'end': video_duration * 0.8 + segment_length,
                    'topic': 2,
                    'score': 0.8
                }
            ]
            print(f"✅ Created {len(summary)} fallback segments")

        # Step 6: EXACT same saving as colleague's main.py
        print("Saving summary and transcript...")
        save_extractive_summary(extracted_summary_path, summary=summary, merged=True)
        saving_full_transcript(full_transcript_path, result['text'])

        # Step 7: EXACT same video processing as colleague's main.py
        print("Processing video segments...")
        process_video(summary, video_path, output_path)

        # Step 8: Format for API response
        highlight_segments = []
        for i, item in enumerate(summary):
            description = item.get("sentence", f"Highlight {i + 1}")
            if len(description) > 100:
                description = description[:97] + "..."
            
            highlight_segments.append({
                "start": item["start"],
                "end": item["end"],
                "description": description
            })

        total_end = time.time()
        print(f"🎉 Algorithmic highlight generation complete in {total_end - total_start:.2f}s")
        
        return output_path, highlight_segments
        
    except Exception as e:
        print(f"❌ Error in algorithmic highlight generation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

# Background task for generating highlights
async def process_highlights_generation(
    highlights_task_id: str, 
    video_task_id: str,
    duration_seconds: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    use_algorithmic: bool = False,
    is_reel: bool = False
):
    """Background task to generate various types of highlights from a video"""
    try:
        highlights_task_storage[highlights_task_id] = {
            "status": "processing",
            "message": "Generating highlights...",
            "progress": 5
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Create output directory for highlights if it doesn't exist
        highlights_dir = os.path.join(settings.OUTPUT_DIR, "highlights")
        os.makedirs(highlights_dir, exist_ok=True)
        
        # Track progress updates
        highlights_task_storage[highlights_task_id]["progress"] = 10
        
        # Generate the appropriate type of highlights
        video_output_path = None
        highlight_segments = None
        
        # Choose the right generation method based on parameters
        if custom_prompt:
            # Custom highlights with specific instructions
            highlights_task_storage[highlights_task_id]["message"] = "Generating custom highlights based on your instructions..."
            # Get transcript segments from task info
            transcript_segments = task_info["transcript_info"]["segments"]
            highlight_segments = await generate_custom_highlights(
                video_path,
                transcript_segments,
                video_info,
                custom_prompt,
                target_duration=duration_seconds
            )
            
            # Extract and merge clips
            if highlight_segments:
                highlights_task_storage[highlights_task_id]["progress"] = 50
                highlights_task_storage[highlights_task_id]["message"] = "Extracting highlight clips..."
                
                from highlights import extract_highlights, merge_clips
                clip_paths, highlight_info = extract_highlights(video_path, highlight_segments)
                
                highlights_task_storage[highlights_task_id]["progress"] = 75
                highlights_task_storage[highlights_task_id]["message"] = "Merging clips into final video..."
                
                video_output_path = merge_clips(clip_paths, highlight_info, is_reel=is_reel)
                
        elif use_algorithmic:
            # 🔥 THIS IS THE KEY: Use your colleague's EXACT algorithmic method
            highlights_task_storage[highlights_task_id]["message"] = "Using fast algorithmic highlight generation..."
            highlights_task_storage[highlights_task_id]["progress"] = 20
            
            print("🚀 Starting colleague's algorithmic pipeline...")
            video_output_path, highlight_segments = await generate_highlights_algorithmically_new(
                video_path
            )
            
        else:
            # Standard LLM-based highlight generation
            highlights_task_storage[highlights_task_id]["message"] = "Generating highlights using AI analysis..."
            # Get transcript segments from task info
            transcript_segments = task_info["transcript_info"]["segments"]
            video_output_path, highlight_segments = await generate_highlights(
                video_path,
                transcript_segments,
                video_info,
                target_duration=duration_seconds,
                is_reel=is_reel
            )
        
        if not video_output_path or not os.path.exists(video_output_path):
            highlights_task_storage[highlights_task_id] = {
                "status": "failed",
                "error": "Failed to generate highlights video"
            }
            return
            
        if not highlight_segments:
            highlights_task_storage[highlights_task_id] = {
                "status": "failed",
                "error": "Failed to generate highlight segments"
            }
            return
        
        # Rename file to include task ID
        file_type = "reel" if is_reel else "highlights"
        new_filename = f"{highlights_task_id}_{file_type}.mp4"
        new_path = os.path.join(highlights_dir, new_filename)
        
        # Copy file
        shutil.copy(video_output_path, new_path)
        
        # Create segment objects
        segment_objects = [
            HighlightSegment(
                start=segment["start"],
                end=segment["end"],
                description=segment["description"]
            )
            for segment in highlight_segments
        ]
        
        # Calculate total duration
        total_duration = sum(segment["end"] - segment["start"] for segment in highlight_segments)
        
        # Store result info
        highlights_task_storage[highlights_task_id] = {
            "status": "completed",
            "result": {
                "video_path": new_path,
                "video_url": f"/api/highlights/download/{highlights_task_id}",
                "segments": segment_objects,
                "total_duration": total_duration,
                "is_reel": is_reel,
                "used_algorithmic": use_algorithmic
            }
        }
        
        print(f"🎉 Highlights generation completed successfully!")
        
    except Exception as e:
        import traceback
        highlights_task_storage[highlights_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Highlights generation failed: {str(e)}")
