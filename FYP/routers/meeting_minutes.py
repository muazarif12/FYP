from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
from meeting_minutes import generate_meeting_minutes, save_meeting_minutes, format_minutes_for_display

router = APIRouter()

# Dictionary to track meeting minutes tasks
meeting_minutes_task_storage = {}

# Data models
class MeetingMinutesRequest(BaseModel):
    task_id: str  # Task ID from video processing
    format: str = "md"  # Output format: "md" (markdown) or "txt" (text)

class MeetingMinutesResponse(BaseModel):
    meeting_minutes_task_id: str
    status: str = "processing"
    message: str

class MeetingMinutesResult(BaseModel):
    title: str
    date: str
    participants: List[str]
    file_url: Optional[str] = None
    preview: str

@router.post("/meeting-minutes", response_model=MeetingMinutesResponse, tags=["Meeting Minutes"])
async def create_meeting_minutes(request: MeetingMinutesRequest, background_tasks: BackgroundTasks):
    """
    Generate structured meeting minutes from a meeting recording.
    
    - **task_id**: Task ID from video processing
    - **format**: Output format (md or txt, default: md)
    
    Returns a task ID to track the meeting minutes generation.
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
    
    # Create a new task ID for this meeting minutes request
    meeting_minutes_task_id = str(uuid.uuid4())
    
    try:
        # Start meeting minutes generation in background
        background_tasks.add_task(
            process_meeting_minutes_generation,
            meeting_minutes_task_id,
            request.task_id,
            request.format
        )
        
        # Initialize task status
        meeting_minutes_task_storage[meeting_minutes_task_id] = {
            "status": "processing",
            "message": "Generating meeting minutes..."
        }
        
        return MeetingMinutesResponse(
            meeting_minutes_task_id=meeting_minutes_task_id,
            status="processing",
            message="Your meeting minutes are being generated. Use the meeting_minutes_task_id to check status and get results."
        )
    
    except Exception as e:
        meeting_minutes_task_storage[meeting_minutes_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting meeting minutes generation: {str(e)}")

@router.get("/meeting-minutes/status/{meeting_minutes_task_id}", tags=["Meeting Minutes"])
async def check_meeting_minutes_status(meeting_minutes_task_id: str):
    """
    Check the status of a meeting minutes generation task.
    
    - **meeting_minutes_task_id**: Meeting minutes task ID to check
    
    Returns the current status of the meeting minutes generation.
    """
    if meeting_minutes_task_id not in meeting_minutes_task_storage:
        raise HTTPException(status_code=404, detail=f"Meeting minutes task {meeting_minutes_task_id} not found")
    
    return meeting_minutes_task_storage[meeting_minutes_task_id]

@router.get("/meeting-minutes/result/{meeting_minutes_task_id}", response_model=MeetingMinutesResult, tags=["Meeting Minutes"])
async def get_meeting_minutes_result(meeting_minutes_task_id: str):
    """
    Get the results of a completed meeting minutes generation task.
    
    - **meeting_minutes_task_id**: Meeting minutes task ID
    
    Returns information about the generated meeting minutes including download URL and preview.
    """
    if meeting_minutes_task_id not in meeting_minutes_task_storage:
        raise HTTPException(status_code=404, detail=f"Meeting minutes task {meeting_minutes_task_id} not found")
    
    task_info = meeting_minutes_task_storage[meeting_minutes_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Meeting minutes task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info:
        raise HTTPException(status_code=500, detail="No result available")
    
    return MeetingMinutesResult(**task_info["result"])

@router.get("/meeting-minutes/download/{meeting_minutes_task_id}", tags=["Meeting Minutes"])
async def download_meeting_minutes(meeting_minutes_task_id: str):
    """
    Download the meeting minutes file.
    
    - **meeting_minutes_task_id**: Meeting minutes task ID
    
    Returns the meeting minutes file (markdown or text).
    """
    if meeting_minutes_task_id not in meeting_minutes_task_storage:
        raise HTTPException(status_code=404, detail=f"Meeting minutes task {meeting_minutes_task_id} not found")
    
    task_info = meeting_minutes_task_storage[meeting_minutes_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Meeting minutes task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "file_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No meeting minutes file available")
    
    file_path = task_info["result"]["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Meeting minutes file not found")
    
    # Determine media type based on file extension
    media_type = "text/markdown" if file_path.endswith(".md") else "text/plain"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=os.path.basename(file_path)
    )

# Background task for generating meeting minutes
async def process_meeting_minutes_generation(meeting_minutes_task_id: str, video_task_id: str, output_format: str):
    """Background task to generate meeting minutes"""
    try:
        meeting_minutes_task_storage[meeting_minutes_task_id] = {
            "status": "processing",
            "message": "Generating meeting minutes..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        transcript_segments = task_info["transcript_info"]["segments"]
        full_text = task_info["transcript_info"]["full_text"]
        detected_language = task_info["transcript_info"]["language"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Create output directory for meeting minutes if it doesn't exist
        meetings_dir = os.path.join(settings.OUTPUT_DIR, "meetings")
        os.makedirs(meetings_dir, exist_ok=True)
        
        # Generate meeting minutes
        meeting_minutes_data = await generate_meeting_minutes(
            transcript_segments,
            video_info,
            detected_language=detected_language,
            timestamped_transcript=full_text
        )
        
        if not meeting_minutes_data:
            meeting_minutes_task_storage[meeting_minutes_task_id] = {
                "status": "failed",
                "error": "Failed to generate meeting minutes"
            }
            return
        
        # Save to file
        file_path = await save_meeting_minutes(meeting_minutes_data, format=output_format)
        
        if not file_path or not os.path.exists(file_path):
            meeting_minutes_task_storage[meeting_minutes_task_id] = {
                "status": "failed",
                "error": "Failed to save meeting minutes to file"
            }
            return
        
        # Rename and move file to include task ID
        new_filename = f"{meeting_minutes_task_id}_meeting_minutes.{output_format}"
        new_path = os.path.join(meetings_dir, new_filename)
        
        # Copy file to new location
        shutil.copy(file_path, new_path)
        
        # Format minutes for preview display
        minutes_preview = format_minutes_for_display(meeting_minutes_data)
        
        # Extract key information for the result
        title = meeting_minutes_data.get("title", "Meeting Minutes")
        date = meeting_minutes_data.get("date", "Not specified")
        participants = meeting_minutes_data.get("participants", [])
        
        # Store result info
        meeting_minutes_task_storage[meeting_minutes_task_id] = {
            "status": "completed",
            "result": {
                "title": title,
                "date": date,
                "participants": participants,
                "file_path": new_path,
                "file_url": f"/api/meeting-minutes/download/{meeting_minutes_task_id}",
                "preview": minutes_preview[:1000] + "..." if len(minutes_preview) > 1000 else minutes_preview
            }
        }
        
    except Exception as e:
        import traceback
        meeting_minutes_task_storage[meeting_minutes_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }