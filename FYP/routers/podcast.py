from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import asyncio
import uuid
from typing import Optional, List

from core.config import settings
from core.models import (
    PodcastRequest,
    PodcastResponse,
    ProcessingStatusResponse
)

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from podcast_integration import generate_podcast
from podcast_generator import generate_podcast_script, save_podcast_script
from gtts_audio_generator import generate_podcast_audio_with_gtts
from downloader import download_video

router = APIRouter()

# Dictionary to track background tasks
task_status = {}

async def process_podcast_generation(
    task_id: str,
    video_path: str,
    transcript_segments: List,
    video_info: dict,
    custom_style: Optional[str] = None,
    detected_language: str = "en"
):
    """Background task to process podcast generation"""
    try:
        task_status[task_id] = {"status": "processing", "progress": 0}
        
        # Update progress
        task_status[task_id]["progress"] = 30
        
        # Generate podcast
        podcast_path, podcast_data = await generate_podcast(
            video_path,
            transcript_segments,
            video_info,
            custom_prompt=custom_style,
            detected_language=detected_language
        )
        
        # Update progress
        task_status[task_id]["progress"] = 100
        
        if podcast_path and os.path.exists(podcast_path):
            # Move the podcast file to the podcast directory with the task ID
            filename = os.path.basename(podcast_path)
            new_path = os.path.join(settings.PODCAST_DIR, f"{task_id}_{filename}")
            
            # Also handle the transcript file
            script_path = None
            script_filename = None
            
            # Check if podcast_path is a script or audio file
            if podcast_path.endswith('.txt'):
                # It's a text script
                script_path = podcast_path
                os.rename(script_path, new_path)
            else:
                # It's an audio file
                os.rename(podcast_path, new_path)
                
                # Try to find the associated script
                script_path = os.path.join(settings.PODCAST_DIR, f"{os.path.splitext(filename)[0]}.txt")
                if os.path.exists(script_path):
                    script_filename = f"{task_id}_{os.path.basename(script_path)}"
                    script_new_path = os.path.join(settings.PODCAST_DIR, script_filename)
                    os.rename(script_path, script_new_path)
                    script_path = script_new_path
            
            # Get podcast duration
            duration = podcast_data.get('estimated_duration_minutes', 5) * 60  # Convert to seconds
            
            # Set task as completed
            task_status[task_id] = {
                "status": "completed", 
                "progress": 100,
                "result": {
                    "podcast_url": f"/downloads/podcast/{os.path.basename(new_path)}",
                    "transcript_url": f"/downloads/podcast/{script_filename}" if script_filename else None,
                    "title": podcast_data.get('title', 'Podcast'),
                    "duration": duration,
                    "hosts": podcast_data.get('hosts', ['Host1', 'Host2'])
                }
            }
        else:
            task_status[task_id] = {
                "status": "failed", 
                "progress": 100,
                "error": "Failed to generate podcast"
            }
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }

@router.post("/podcast/generate", response_model=ProcessingStatusResponse, 
             summary="Generate podcast from video")
async def generate_video_podcast(
    request: PodcastRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a podcast (conversation) from a YouTube video or uploaded video.
    
    - **video_id**: ID of the uploaded video (if already uploaded)
    - **youtube_url**: YouTube URL (will be downloaded)
    - **transcript_id**: ID of an existing transcript (if already processed)
    - **style**: Custom style for the podcast (casual, formal, educational, etc.)
    
    Returns a task ID for checking the status of the podcast generation process.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Check that we have a video source
        if not request.video_id and not request.youtube_url:
            raise HTTPException(status_code=400, detail="Either video_id or youtube_url is required")
        
        # Get transcript segments
        transcript_segments = None
        detected_language = "en"
        
        if request.transcript_id:
            # Load transcript from file
            transcript_file = os.path.join(settings.TRANSCRIPTS_DIR, f"{request.transcript_id}_transcript.txt")
            
            if not os.path.exists(transcript_file):
                raise HTTPException(status_code=404, detail=f"Transcript {request.transcript_id} not found")
            
            # Logic to load transcript segments from file
            # This is a placeholder - you'll need to implement this based on your transcript format
            # transcript_segments = load_transcript_segments_from_file(transcript_file)
            
            # Placeholder for now - you'll need to get this from your transcription service
            raise HTTPException(status_code=400, detail="Loading from transcript_id not yet implemented")
        
        # Get the video file
        video_path = None
        video_info = {}
        
        if request.youtube_url:
            # Download from YouTube
            downloaded_file, video_title, video_description, video_id = await download_video(request.youtube_url)
            
            if not downloaded_file:
                raise HTTPException(status_code=400, detail="Failed to download YouTube video")
            
            video_path = downloaded_file
            video_info = {"title": video_title, "description": video_description}
            
            # If we don't have transcript segments already, get them from YouTube
            if not transcript_segments:
                from transcriber import get_youtube_transcript, transcribe_video
                
                # Try YouTube transcript first
                if video_id:
                    transcript_segments, _, detected_language = await get_youtube_transcript(video_id)
                
                # Fall back to Whisper transcription
                if not transcript_segments:
                    transcript_segments, _, detected_language = await transcribe_video(video_path)
                    
                if not transcript_segments:
                    raise HTTPException(status_code=400, detail="Failed to transcribe video")
        
        elif request.video_id:
            # Get video from storage
            video_path = os.path.join(settings.TEMP_DIR, f"{request.video_id}")
            
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")
            
            # If we don't have transcript segments, transcribe the video
            if not transcript_segments:
                from transcriber import transcribe_video
                transcript_segments, _, detected_language = await transcribe_video(video_path)
                
                if not transcript_segments:
                    raise HTTPException(status_code=400, detail="Failed to transcribe video")
        
        # Start processing in background
        background_tasks.add_task(
            process_podcast_generation,
            task_id,
            video_path,
            transcript_segments,
            video_info,
            request.style,
            detected_language
        )
        
        return ProcessingStatusResponse(
            task_id=task_id,
            status="processing",
            progress=0,
            result_url=None
        )
        
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/podcast/status/{task_id}", response_model=ProcessingStatusResponse, 
            summary="Check podcast generation status")
async def check_podcast_status(task_id: str):
    """
    Check the status of a podcast generation task.
    
    - **task_id**: The ID of the task to check
    
    Returns the current status of the podcast generation process.
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    status_info = task_status[task_id]
    
    response = ProcessingStatusResponse(
        task_id=task_id,
        status=status_info["status"],
        progress=status_info["progress"],
        result_url=None,
        error=status_info.get("error")
    )
    
    # If completed, add results URL
    if status_info["status"] == "completed":
        result = status_info.get("result", {})
        response.result_url = result.get("podcast_url")
    
    return response

@router.get("/podcast/result/{task_id}", response_model=PodcastResponse,
           summary="Get podcast generation results")
async def get_podcast_result(task_id: str):
    """
    Get the results of a completed podcast generation task.
    
    - **task_id**: The ID of the completed task
    
    Returns the podcast generation results.
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    status_info = task_status[task_id]
    
    if status_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task is {status_info['status']}, not completed"
        )
    
    if "result" not in status_info:
        raise HTTPException(status_code=500, detail="No result available")
    
    return PodcastResponse(**status_info["result"])

@router.get("/podcast/{filename}", summary="Download a podcast file")
async def get_podcast_file(filename: str):
    """
    Download a specific podcast file (audio or transcript).
    
    - **filename**: The name of the podcast file to download
    
    Returns the podcast file.
    """
    file_path = os.path.join(settings.PODCAST_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Podcast file {filename} not found")
    
    # Determine media type based on file extension
    media_type = "audio/mpeg" if filename.endswith((".mp3", ".mpeg")) else "text/plain"
    
    return FileResponse(
        file_path, 
        media_type=media_type, 
        filename=filename
    )