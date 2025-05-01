from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid
import shutil

from core.config import settings

# Import from utils router to access task storage
from routers.utils import task_storage

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from subtitling import create_english_subtitles, adjust_subtitle_timing_by_offset

router = APIRouter()

# Dictionary to track subtitle tasks
subtitle_task_storage = {}

# Data models
class SubtitleRequest(BaseModel):
    task_id: str  # Task ID from video processing
    subtitle_format: str = "srt"  # Default format (srt or vtt)

class SubtitleResponse(BaseModel):
    subtitle_task_id: str
    status: str = "processing"
    message: str

class SubtitleAdjustRequest(BaseModel):
    subtitle_task_id: str
    offset_seconds: float  # Positive to delay, negative to advance

@router.post("/subtitles", response_model=SubtitleResponse, tags=["Subtitles"])
async def create_subtitles(request: SubtitleRequest, background_tasks: BackgroundTasks):
    """
    Generate English subtitles for the video.
    
    - **task_id**: Task ID from video processing
    - **subtitle_format**: Format of the subtitle file (srt or vtt)
    
    Returns a task ID to track the subtitle generation.
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
    
    # Create a new task ID for this subtitle request
    subtitle_task_id = str(uuid.uuid4())
    
    try:
        # Start subtitle generation in background
        background_tasks.add_task(
            process_subtitle_generation,
            subtitle_task_id,
            request.task_id,
            request.subtitle_format
        )
        
        # Initialize task status
        subtitle_task_storage[subtitle_task_id] = {
            "status": "processing",
            "message": "Generating subtitles..."
        }
        
        return SubtitleResponse(
            subtitle_task_id=subtitle_task_id,
            status="processing",
            message="Your subtitles are being generated. Use the subtitle_task_id to check status and get results."
        )
    
    except Exception as e:
        subtitle_task_storage[subtitle_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting subtitle generation: {str(e)}")

@router.get("/subtitles/status/{subtitle_task_id}", tags=["Subtitles"])
async def check_subtitle_status(subtitle_task_id: str):
    """
    Check the status of a subtitle generation task.
    
    - **subtitle_task_id**: Subtitle task ID to check
    
    Returns the current status of the subtitle generation.
    """
    if subtitle_task_id not in subtitle_task_storage:
        raise HTTPException(status_code=404, detail=f"Subtitle task {subtitle_task_id} not found")
    
    return subtitle_task_storage[subtitle_task_id]

@router.get("/subtitles/download/{subtitle_task_id}", tags=["Subtitles"])
async def download_subtitles(subtitle_task_id: str):
    """
    Download the subtitle file.
    
    - **subtitle_task_id**: Subtitle task ID
    
    Returns the subtitle file.
    """
    if subtitle_task_id not in subtitle_task_storage:
        raise HTTPException(status_code=404, detail=f"Subtitle task {subtitle_task_id} not found")
    
    task_info = subtitle_task_storage[subtitle_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Subtitle task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "subtitle_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No subtitle file available for download")
    
    subtitle_path = task_info["result"]["subtitle_path"]
    
    if not os.path.exists(subtitle_path):
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    
    # Determine media type based on extension
    media_type = "application/x-subrip" if subtitle_path.endswith(".srt") else "text/vtt"
    
    return FileResponse(
        subtitle_path,
        media_type=media_type,
        filename=os.path.basename(subtitle_path)
    )

@router.get("/subtitles/video/{subtitle_task_id}", tags=["Subtitles"])
async def download_subtitled_video(subtitle_task_id: str):
    """
    Download the video with embedded subtitles.
    
    - **subtitle_task_id**: Subtitle task ID
    
    Returns the video file with embedded subtitles.
    """
    if subtitle_task_id not in subtitle_task_storage:
        raise HTTPException(status_code=404, detail=f"Subtitle task {subtitle_task_id} not found")
    
    task_info = subtitle_task_storage[subtitle_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Subtitle task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "video_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No subtitled video available for download")
    
    video_path = task_info["result"]["video_path"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Subtitled video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )

@router.post("/subtitles/adjust", response_model=SubtitleResponse, tags=["Subtitles"])
async def adjust_subtitle_timing(request: SubtitleAdjustRequest, background_tasks: BackgroundTasks):
    """
    Adjust the timing of an existing subtitle file.
    
    - **subtitle_task_id**: ID of the existing subtitle task
    - **offset_seconds**: Number of seconds to shift (positive = delay, negative = advance)
    
    Returns a new task ID for the adjusted subtitles.
    """
    # Verify that the subtitle task exists and is completed
    if request.subtitle_task_id not in subtitle_task_storage:
        raise HTTPException(status_code=404, detail=f"Subtitle task {request.subtitle_task_id} not found")
    
    source_task_info = subtitle_task_storage[request.subtitle_task_id]
    
    if source_task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Source subtitle task not complete. Current status: {source_task_info['status']}"
        )
    
    if "result" not in source_task_info or "subtitle_path" not in source_task_info["result"]:
        raise HTTPException(status_code=404, detail="No subtitle file available to adjust")
    
    # Create a new task ID for this adjustment
    new_subtitle_task_id = str(uuid.uuid4())
    
    try:
        # Start subtitle adjustment in background
        background_tasks.add_task(
            process_subtitle_adjustment,
            new_subtitle_task_id,
            request.subtitle_task_id,
            request.offset_seconds
        )
        
        # Initialize task status
        subtitle_task_storage[new_subtitle_task_id] = {
            "status": "processing",
            "message": "Adjusting subtitle timing..."
        }
        
        return SubtitleResponse(
            subtitle_task_id=new_subtitle_task_id,
            status="processing",
            message="Your subtitles are being adjusted. Use the new subtitle_task_id to check status and get results."
        )
    
    except Exception as e:
        subtitle_task_storage[new_subtitle_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting subtitle adjustment: {str(e)}")

# Background task for generating subtitles
async def process_subtitle_generation(subtitle_task_id: str, video_task_id: str, subtitle_format: str):
    """Background task to generate subtitles for a video"""
    try:
        subtitle_task_storage[subtitle_task_id] = {
            "status": "processing",
            "message": "Generating subtitles..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        transcript_segments = task_info["transcript_info"]["segments"]
        detected_language = task_info["transcript_info"]["language"]
        
        # Create output directory for subtitles if it doesn't exist
        subtitles_dir = os.path.join(settings.OUTPUT_DIR, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        
        # Generate subtitles
        subtitled_video_path, subtitle_path, stats = await create_english_subtitles(
            video_path,
            transcript_segments,
            detected_language,
            subtitles_dir
        )
        
        if not subtitle_path:
            subtitle_task_storage[subtitle_task_id] = {
                "status": "failed",
                "error": "Failed to generate subtitles"
            }
            return
        
        # Rename files to include task ID
        new_subtitle_filename = f"{subtitle_task_id}_subtitles.{subtitle_format}"
        new_subtitle_path = os.path.join(subtitles_dir, new_subtitle_filename)
        
        # Copy subtitle file
        shutil.copy(subtitle_path, new_subtitle_path)
        
        # If video with embedded subtitles was created, rename it too
        new_video_path = None
        if subtitled_video_path and os.path.exists(subtitled_video_path):
            new_video_filename = f"{subtitle_task_id}_subtitled.mp4"
            new_video_path = os.path.join(subtitles_dir, new_video_filename)
            shutil.copy(subtitled_video_path, new_video_path)
        
        # Store result info
        subtitle_task_storage[subtitle_task_id] = {
            "status": "completed",
            "result": {
                "subtitle_path": new_subtitle_path,
                "video_path": new_video_path,
                "original_language": detected_language,
                "subtitle_format": subtitle_format,
                "subtitle_url": f"/api/subtitles/download/{subtitle_task_id}",
                "video_url": f"/api/subtitles/video/{subtitle_task_id}" if new_video_path else None,
                "stats": stats
            }
        }
        
    except Exception as e:
        import traceback
        subtitle_task_storage[subtitle_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Background task for adjusting subtitle timing
async def process_subtitle_adjustment(new_subtitle_task_id: str, source_subtitle_task_id: str, offset_seconds: float):
    """Background task to adjust timing of existing subtitles"""
    try:
        subtitle_task_storage[new_subtitle_task_id] = {
            "status": "processing",
            "message": "Adjusting subtitle timing..."
        }
        
        # Get source subtitle info
        source_task_info = subtitle_task_storage[source_subtitle_task_id]
        source_subtitle_path = source_task_info["result"]["subtitle_path"]
        subtitle_format = source_task_info["result"]["subtitle_format"]
        
        # Create output directory for subtitles if it doesn't exist
        subtitles_dir = os.path.join(settings.OUTPUT_DIR, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        
        # Adjust subtitle timing
        adjusted_subtitle_path = await adjust_subtitle_timing_by_offset(
            source_subtitle_path,
            offset_seconds,
            output_dir=subtitles_dir
        )
        
        if not adjusted_subtitle_path:
            subtitle_task_storage[new_subtitle_task_id] = {
                "status": "failed",
                "error": "Failed to adjust subtitle timing"
            }
            return
        
        # Rename file to include task ID
        new_subtitle_filename = f"{new_subtitle_task_id}_adjusted.{subtitle_format}"
        new_subtitle_path = os.path.join(subtitles_dir, new_subtitle_filename)
        
        # Copy file
        shutil.copy(adjusted_subtitle_path, new_subtitle_path)
        
        # Store result info
        subtitle_task_storage[new_subtitle_task_id] = {
            "status": "completed",
            "result": {
                "subtitle_path": new_subtitle_path,
                "subtitle_format": subtitle_format,
                "original_subtitle_task_id": source_subtitle_task_id,
                "offset_applied": offset_seconds,
                "subtitle_url": f"/api/subtitles/download/{new_subtitle_task_id}"
            }
        }
        
    except Exception as e:
        import traceback
        subtitle_task_storage[new_subtitle_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }