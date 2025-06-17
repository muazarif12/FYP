import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid

from core.config import settings

# Import from utils router to access task storage
from routers.utils import task_storage

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from podcast_integration import generate_podcast

router = APIRouter()

# Dictionary to track podcast tasks
podcast_task_storage = {}

# Data models
class PodcastRequest(BaseModel):
    task_id: str  # Task ID from video processing
    style: Optional[str] = None  # Optional podcast style

class PodcastResponse(BaseModel):
    podcast_task_id: str
    status: str = "processing"
    message: str

class PodcastResult(BaseModel):
    title: str
    audio_url: Optional[str] = None
    transcript_url: Optional[str] = None
    duration_minutes: float
    hosts: List[str]

@router.post("/podcast", response_model=PodcastResponse, tags=["Podcast"])
async def create_podcast(request: PodcastRequest, background_tasks: BackgroundTasks):
    """
    Generate a conversational podcast from the video content.
    
    - **task_id**: Task ID from video processing
    - **style**: Optional style for the podcast (casual, educational, formal, etc.)
    
    Returns a task ID to track the podcast generation.
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
    
    # Create a new task ID for this podcast request
    podcast_task_id = str(uuid.uuid4())
    
    try:
        # Start podcast generation in background
        background_tasks.add_task(
            process_podcast_generation,
            podcast_task_id,
            request.task_id,
            request.style
        )
        
        # Initialize task status
        podcast_task_storage[podcast_task_id] = {
            "status": "processing",
            "message": "Generating podcast..."
        }
        
        return PodcastResponse(
            podcast_task_id=podcast_task_id,
            status="processing",
            message="Your podcast is being generated. Use the podcast_task_id to check status and get results."
        )
    
    except Exception as e:
        podcast_task_storage[podcast_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting podcast generation: {str(e)}")

@router.get("/podcast/status/{podcast_task_id}", tags=["Podcast"])
async def check_podcast_status(podcast_task_id: str):
    """
    Check the status of a podcast generation task.
    
    - **podcast_task_id**: Podcast task ID to check
    
    Returns the current status of the podcast generation.
    """
    if podcast_task_id not in podcast_task_storage:
        raise HTTPException(status_code=404, detail=f"Podcast task {podcast_task_id} not found")
    
    return podcast_task_storage[podcast_task_id]

@router.get("/podcast/result/{podcast_task_id}", response_model=PodcastResult, tags=["Podcast"])
async def get_podcast_result(podcast_task_id: str):
    """
    Get the results of a completed podcast generation task.
    
    - **podcast_task_id**: Podcast task ID
    
    Returns information about the generated podcast including URLs to audio and transcript.
    """
    if podcast_task_id not in podcast_task_storage:
        raise HTTPException(status_code=404, detail=f"Podcast task {podcast_task_id} not found")
    
    task_info = podcast_task_storage[podcast_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Podcast task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info:
        raise HTTPException(status_code=500, detail="No result available")
    
    return PodcastResult(**task_info["result"])

@router.get("/podcast/audio/{podcast_task_id}", tags=["Podcast"])
async def get_podcast_audio(podcast_task_id: str):
    """
    Download the podcast audio.
    
    - **podcast_task_id**: Podcast task ID
    
    Returns the podcast audio file.
    """
    if podcast_task_id not in podcast_task_storage:
        raise HTTPException(status_code=404, detail=f"Podcast task {podcast_task_id} not found")
    
    task_info = podcast_task_storage[podcast_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Podcast task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "audio_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No audio available for this podcast")
    
    audio_path = task_info["result"]["audio_path"]
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"podcast_{podcast_task_id}.mp3"
    )

@router.get("/podcast/transcript/{podcast_task_id}", tags=["Podcast"])
async def get_podcast_transcript(podcast_task_id: str):
    """
    Download the podcast transcript.
    
    - **podcast_task_id**: Podcast task ID
    
    Returns the podcast transcript file.
    """
    if podcast_task_id not in podcast_task_storage:
        raise HTTPException(status_code=404, detail=f"Podcast task {podcast_task_id} not found")
    
    task_info = podcast_task_storage[podcast_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Podcast task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "transcript_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No transcript available for this podcast")
    
    transcript_path = task_info["result"]["transcript_path"]
    
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    return FileResponse(
        transcript_path,
        media_type="text/plain",
        filename=f"podcast_transcript_{podcast_task_id}.txt"
    )

# Background task for processing podcast generation
async def process_podcast_generation(podcast_task_id: str, video_task_id: str, style: Optional[str] = None):
    """Background task to generate podcast from video"""
    try:
        podcast_task_storage[podcast_task_id] = {
            "status": "processing",
            "message": "Generating podcast..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        transcript_segments = task_info["transcript_info"]["segments"]
        detected_language = task_info["transcript_info"]["language"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Create output directory for podcasts if it doesn't exist
        podcasts_dir = os.path.join(settings.OUTPUT_DIR, "podcast")
        os.makedirs(podcasts_dir, exist_ok=True)
        
        # Generate podcast
        podcast_path, podcast_data = await generate_podcast(
            video_path,
            transcript_segments,
            video_info,
            custom_prompt=style,
            detected_language=detected_language
        )
        
        if not podcast_path:
            podcast_task_storage[podcast_task_id] = {
                "status": "failed",
                "error": "Failed to generate podcast"
            }
            return
        
        # Determine if it's an audio file or just a transcript
        is_audio = podcast_path.endswith(('.mp3', '.wav', '.ogg', '.m4a'))
        
        # Rename and move files to permanent location
        audio_path = None
        transcript_path = None
        
        if is_audio:
            # It's an audio file
            audio_filename = f"{podcast_task_id}_podcast.mp3"
            audio_path = os.path.join(podcasts_dir, audio_filename)
            
            # Move the file
            shutil.copy(podcast_path, audio_path)
            
            # Try to find associated transcript
            script_base = os.path.splitext(podcast_path)[0]
            if os.path.exists(f"{script_base}.txt"):
                transcript_filename = f"{podcast_task_id}_transcript.txt"
                transcript_path = os.path.join(podcasts_dir, transcript_filename)
                shutil.copy(f"{script_base}.txt", transcript_path)
        else:
            # It's just a transcript
            transcript_filename = f"{podcast_task_id}_transcript.txt"
            transcript_path = os.path.join(podcasts_dir, transcript_filename)
            
            # Move the file
            shutil.copy(podcast_path, transcript_path)
        
        # Calculate duration estimate
        if 'estimated_duration_minutes' in podcast_data:
            duration_minutes = float(podcast_data['estimated_duration_minutes'])
        else:
            # Estimate based on word count if audio not generated
            script = podcast_data.get('script', [])
            total_words = sum(len(line.get('text', '').split()) for line in script)
            # Average speaking pace: ~150 words per minute
            duration_minutes = total_words / 150
        
        # Store the result
        result = {
            "title": podcast_data.get('title', 'Podcast about ' + video_info["title"]),
            "hosts": podcast_data.get('hosts', ['Host1', 'Host2']),
            "duration_minutes": duration_minutes,
            "audio_url": f"/api/podcast/audio/{podcast_task_id}" if audio_path else None,
            "transcript_url": f"/api/podcast/transcript/{podcast_task_id}" if transcript_path else None,
            "audio_path": audio_path,  # For internal use
            "transcript_path": transcript_path  # For internal use
        }
        
        podcast_task_storage[podcast_task_id] = {
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        import traceback
        podcast_task_storage[podcast_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }