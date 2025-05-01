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
from highlights import generate_highlights, generate_custom_highlights
from algorithmic_highlights import generate_highlights_algorithmically

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
            "message": "Generating highlights..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        transcript_segments = task_info["transcript_info"]["segments"]
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
            # Fast algorithmic method
            highlights_task_storage[highlights_task_id]["message"] = "Using fast algorithmic highlight generation..."
            video_output_path, highlight_segments = await generate_highlights_algorithmically(
                video_path,
                transcript_segments,
                video_info,
                target_duration=duration_seconds,
                is_reel=is_reel
            )
            
        else:
            # Standard LLM-based highlight generation
            highlights_task_storage[highlights_task_id]["message"] = "Generating highlights using AI analysis..."
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
        
    except Exception as e:
        import traceback
        highlights_task_storage[highlights_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }