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
from dubbing import create_english_dub

router = APIRouter()

# Dictionary to track dubbing tasks
dubbing_task_storage = {}

# Data models
class DubbingRequest(BaseModel):
    task_id: str  # Task ID from video processing

class DubbingResponse(BaseModel):
    dubbing_task_id: str
    status: str = "processing"
    message: str

@router.post("/dub", response_model=DubbingResponse, tags=["Dubbing"])
async def create_english_dubbing(request: DubbingRequest, background_tasks: BackgroundTasks):
    """
    Generate English dubbed version of a non-English video.
    
    - **task_id**: Task ID from video processing
    
    Returns a task ID to track the dubbing generation process.
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
    
    # Check if the video is already in English
    detected_language = task_info["transcript_info"]["language"]
    if detected_language == "en":
        raise HTTPException(
            status_code=400, 
            detail="Video is already in English, dubbing not required"
        )
    
    # Create a new task ID for this dubbing request
    dubbing_task_id = str(uuid.uuid4())
    
    try:
        # Start dubbing generation in background
        background_tasks.add_task(
            process_dubbing_generation,
            dubbing_task_id,
            request.task_id
        )
        
        # Initialize task status
        dubbing_task_storage[dubbing_task_id] = {
            "status": "processing",
            "message": "Generating English dub..."
        }
        
        return DubbingResponse(
            dubbing_task_id=dubbing_task_id,
            status="processing",
            message="Your English dub is being generated. Use the dubbing_task_id to check status and get results."
        )
    
    except Exception as e:
        dubbing_task_storage[dubbing_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting English dubbing: {str(e)}")

@router.get("/dub/status/{dubbing_task_id}", tags=["Dubbing"])
async def check_dubbing_status(dubbing_task_id: str):
    """
    Check the status of a dubbing generation task.
    
    - **dubbing_task_id**: Dubbing task ID to check
    
    Returns the current status of the dubbing generation.
    """
    if dubbing_task_id not in dubbing_task_storage:
        raise HTTPException(status_code=404, detail=f"Dubbing task {dubbing_task_id} not found")
    
    return dubbing_task_storage[dubbing_task_id]

@router.get("/dub/download/{dubbing_task_id}", tags=["Dubbing"])
async def download_dubbed_video(dubbing_task_id: str):
    """
    Download the English dubbed video.
    
    - **dubbing_task_id**: Dubbing task ID
    
    Returns the dubbed video file.
    """
    if dubbing_task_id not in dubbing_task_storage:
        raise HTTPException(status_code=404, detail=f"Dubbing task {dubbing_task_id} not found")
    
    task_info = dubbing_task_storage[dubbing_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Dubbing task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "dubbed_video_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No dubbed video available")
    
    dubbed_video_path = task_info["result"]["dubbed_video_path"]
    
    if not os.path.exists(dubbed_video_path):
        raise HTTPException(status_code=404, detail="Dubbed video file not found")
    
    return FileResponse(
        dubbed_video_path,
        media_type="video/mp4",
        filename=os.path.basename(dubbed_video_path)
    )

# Background task for generating English dub
async def process_dubbing_generation(dubbing_task_id: str, video_task_id: str):
    """Background task to generate English dub for a non-English video"""
    try:
        dubbing_task_storage[dubbing_task_id] = {
            "status": "processing",
            "message": "Generating English dub..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        transcript_segments = task_info["transcript_info"]["segments"]
        detected_language = task_info["transcript_info"]["language"]
        
        # Create output directory for dubbing if it doesn't exist
        dubbing_dir = os.path.join(settings.OUTPUT_DIR, "dubbing")
        os.makedirs(dubbing_dir, exist_ok=True)
        
        # Generate English dub
        dubbed_video_path, stats = await create_english_dub(
            video_path,
            transcript_segments,
            detected_language,
            dubbing_dir
        )
        
        if not dubbed_video_path:
            error_message = "Failed to generate English dub"
            if isinstance(stats, str):
                error_message = stats
                
            dubbing_task_storage[dubbing_task_id] = {
                "status": "failed",
                "error": error_message
            }
            return
        
        # Rename file to include task ID
        new_filename = f"{dubbing_task_id}_dubbed.mp4"
        new_path = os.path.join(dubbing_dir, new_filename)
        
        # Copy file
        shutil.copy(dubbed_video_path, new_path)
        
        # Store result info
        dubbing_task_storage[dubbing_task_id] = {
            "status": "completed",
            "result": {
                "dubbed_video_path": new_path,
                "original_language": detected_language,
                "download_url": f"/api/dub/download/{dubbing_task_id}",
                "stats": stats
            }
        }
        
    except Exception as e:
        import traceback
        dubbing_task_storage[dubbing_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }