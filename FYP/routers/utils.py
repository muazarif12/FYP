from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import os
import uuid
import asyncio

from core.config import settings

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from downloader import download_video
from transcriber import get_youtube_transcript, transcribe_video

router = APIRouter()

# Dictionary to track tasks
task_storage = {}

# Data models
class YouTubeInput(BaseModel):
    url: HttpUrl
    
class TaskResponse(BaseModel):
    task_id: str
    status: str = "processing"
    message: str

@router.post("/process-youtube", response_model=TaskResponse, tags=["Video Input"])
async def process_youtube_video(input_data: YouTubeInput, background_tasks: BackgroundTasks):
    """
    Process a YouTube video by URL. This is the primary entry point for YouTube videos.
    
    - **url**: YouTube video URL
    
    Returns a task ID to track the processing status.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Start video processing in background
        background_tasks.add_task(
            download_and_process_youtube, 
            task_id, 
            str(input_data.url)
        )
        
        return TaskResponse(
            task_id=task_id,
            status="processing",
            message="YouTube video is being processed. Use this task_id to track status and access results."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

@router.post("/upload-video", response_model=TaskResponse, tags=["Video Input"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and process a video file. This is the primary entry point for uploading videos.
    
    - **file**: Video file to upload and process
    
    Returns a task ID to track the processing status.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                buffer.write(chunk)
        
        # Start processing in background
        background_tasks.add_task(
            process_uploaded_video,
            task_id,
            file_path
        )
        
        return TaskResponse(
            task_id=task_id,
            status="processing",
            message="Video uploaded and is being processed. Use this task_id to track status and access results."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/status/{task_id}", tags=["Video Input"])
async def get_task_status(task_id: str):
    """
    Check the status of a processing task.
    
    - **task_id**: Task ID to check
    
    Returns the current status and available results.
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return task_storage[task_id]

# Background task functions
async def download_and_process_youtube(task_id: str, youtube_url: str):
    """Background task to download and process a YouTube video"""
    try:
        # Update status
        task_storage[task_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Downloading YouTube video"
        }
        
        # Download video
        video_path, video_title, video_description, youtube_id = await download_video(youtube_url)
        
        if not video_path:
            task_storage[task_id] = {
                "status": "failed",
                "message": "Failed to download YouTube video"
            }
            return
        
        # Update status
        task_storage[task_id] = {
            "status": "transcribing",
            "progress": 40,
            "message": "Video downloaded, generating transcript"
        }
        
        # Try to get YouTube transcript first
        transcript_segments = None
        full_transcript = None
        detected_language = None
        
        if youtube_id:
            transcript_segments, full_transcript, detected_language = await get_youtube_transcript(youtube_id)
        
        # If no YouTube transcript, transcribe with Whisper
        if not transcript_segments:
            transcript_segments, full_transcript, detected_language = await transcribe_video(video_path)
        
        if not transcript_segments:
            task_storage[task_id] = {
                "status": "failed",
                "message": "Failed to generate transcript"
            }
            return
        
        # Store the video and transcript info
        task_storage[task_id] = {
            "status": "completed",
            "progress": 100,
            "video_info": {
                "path": video_path,
                "title": video_title,
                "description": video_description,
                "youtube_id": youtube_id
            },
            "transcript_info": {
                "segments": transcript_segments,
                "full_text": full_transcript,
                "language": detected_language
            },
            "message": "Video processing completed successfully"
        }
        
    except Exception as e:
        task_storage[task_id] = {
            "status": "failed",
            "message": f"Error processing video: {str(e)}"
        }

async def process_uploaded_video(task_id: str, file_path: str):
    """Background task to process an uploaded video"""
    try:
        # Update status
        task_storage[task_id] = {
            "status": "transcribing",
            "progress": 30,
            "message": "Video uploaded, generating transcript"
        }
        
        # Get file info
        file_name = os.path.basename(file_path)
        video_title = file_name
        
        # Generate transcript
        transcript_segments, full_transcript, detected_language = await transcribe_video(file_path)
        
        if not transcript_segments:
            task_storage[task_id] = {
                "status": "failed",
                "message": "Failed to generate transcript"
            }
            return
        
        # Store the video and transcript info
        task_storage[task_id] = {
            "status": "completed",
            "progress": 100,
            "video_info": {
                "path": file_path,
                "title": video_title,
                "description": "",
                "youtube_id": None
            },
            "transcript_info": {
                "segments": transcript_segments,
                "full_text": full_transcript,
                "language": detected_language
            },
            "message": "Video processing completed successfully"
        }
        
    except Exception as e:
        task_storage[task_id] = {
            "status": "failed",
            "message": f"Error processing video: {str(e)}"
        }