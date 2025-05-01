from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import asyncio
import shutil
import uuid
from typing import Optional, List, Dict, Any

from core.config import settings
from core.models import YouTubeRequest, ProcessingStatusResponse

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from downloader import download_video

router = APIRouter()

# Dictionary to track background tasks
task_status = {}

@router.post("/upload", response_model=ProcessingStatusResponse, summary="Upload a video file")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video file for processing.
    
    - **file**: The video file to upload
    
    Returns a task ID that can be used to access the uploaded video.
    """
    task_id = str(uuid.uuid4())
    
    # Check file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB
    
    # Save the uploaded file to the temp directory
    file_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")
    
    try:
        task_status[task_id] = {"status": "uploading", "progress": 0}
        
        with open(file_path, 'wb') as f:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                task_status[task_id]["progress"] = min(95, int((file_size / (settings.MAX_UPLOAD_SIZE * 1024 * 1024)) * 100))
                
                if file_size > settings.MAX_UPLOAD_SIZE * 1024 * 1024:
                    # Clean up the partial file
                    os.remove(file_path)
                    task_status[task_id] = {
                        "status": "failed", 
                        "progress": 100,
                        "error": f"File too large. Max size is {settings.MAX_UPLOAD_SIZE}MB."
                    }
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE}MB."
                    )
                f.write(chunk)
        
        # File upload complete
        task_status[task_id] = {
            "status": "completed", 
            "progress": 100,
            "result": {
                "video_id": task_id,
                "filename": file.filename,
                "file_size": file_size,
                "file_path": file_path
            }
        }
        
        return ProcessingStatusResponse(
            task_id=task_id,
            status="completed",
            progress=100,
            result_url=f"/api/videos/{task_id}"
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle other exceptions
        if os.path.exists(file_path):
            os.remove(file_path)
            
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-youtube", response_model=ProcessingStatusResponse, summary="Download a YouTube video")
async def download_youtube_video(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks
):
    """
    Download a YouTube video for processing.
    
    - **youtube_url**: The URL of the YouTube video to download
    
    Returns a task ID that can be used to access the downloaded video.
    """
    task_id = str(uuid.uuid4())
    
    try:
        task_status[task_id] = {"status": "downloading", "progress": 0}
        
        # Start background task
        background_tasks.add_task(
            process_youtube_download,
            task_id,
            request.youtube_url
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

async def process_youtube_download(
    task_id: str,
    youtube_url: str
):
    """Background task to download a YouTube video"""
    try:
        task_status[task_id]["progress"] = 10
        
        # Download the video
        downloaded_file, video_title, video_description, video_id = await download_video(youtube_url)
        
        if not downloaded_file or not os.path.exists(downloaded_file):
            task_status[task_id] = {
                "status": "failed", 
                "progress": 100,
                "error": "Failed to download YouTube video"
            }
            return
        
        task_status[task_id]["progress"] = 90
        
        # Move the downloaded file to a more permanent location with the task ID
        file_extension = os.path.splitext(downloaded_file)[1]
        new_file_path = os.path.join(settings.TEMP_DIR, f"{task_id}{file_extension}")
        os.rename(downloaded_file, new_file_path)
        
        # Update task status
        task_status[task_id] = {
            "status": "completed", 
            "progress": 100,
            "result": {
                "video_id": task_id,
                "filename": os.path.basename(new_file_path),
                "title": video_title,
                "description": video_description,
                "youtube_id": video_id,
                "file_path": new_file_path
            }
        }
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }

@router.get("/videos/{video_id}", summary="Get information about a video")
async def get_video_info(video_id: str):
    """
    Get information about an uploaded or downloaded video.
    
    - **video_id**: The ID of the video
    
    Returns information about the video.
    """
    # Check if it's a completed task
    if video_id in task_status and task_status[video_id]["status"] == "completed":
        if "result" in task_status[video_id]:
            result = task_status[video_id]["result"]
            return JSONResponse(content=result)
    
    # Check if the file exists directly
    possible_files = []
    for file in os.listdir(settings.TEMP_DIR):
        if file.startswith(f"{video_id}"):
            possible_files.append(file)
    
    if possible_files:
        # Use the first matching file (should be only one)
        file_path = os.path.join(settings.TEMP_DIR, possible_files[0])
        return JSONResponse(content={
            "video_id": video_id,
            "filename": possible_files[0],
            "file_path": file_path
        })
    
    # Video not found
    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

@router.get("/videos/{video_id}/stream", summary="Stream a video")
async def stream_video(video_id: str):
    """
    Stream a video file.
    
    - **video_id**: The ID of the video to stream
    
    Returns the video file for streaming.
    """
    # Find the video file
    possible_files = []
    for file in os.listdir(settings.TEMP_DIR):
        if file.startswith(f"{video_id}"):
            possible_files.append(file)
    
    if not possible_files:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    # Use the first matching file (should be only one)
    file_path = os.path.join(settings.TEMP_DIR, possible_files[0])
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Video file not found")
    
    # Return the file for streaming
    return FileResponse(
        file_path, 
        media_type="video/mp4",  # Adjust based on file extension if needed
        filename=possible_files[0]
    )

@router.get("/task/{task_id}", response_model=ProcessingStatusResponse, summary="Check task status")
async def check_task_status(task_id: str):
    """
    Check the status of any task.
    
    - **task_id**: The ID of the task to check
    
    Returns the current status of the task.
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
    
    # If completed, add results URL if available
    if status_info["status"] == "completed" and "result" in status_info:
        result = status_info["result"]
        if "video_id" in result:
            response.result_url = f"/api/videos/{result['video_id']}"
    
    return response