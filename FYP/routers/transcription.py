from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import asyncio
import shutil
import uuid
from typing import Optional
import time

from core.config import settings
from core.models import TranscriptionResponse, YouTubeRequest, ProcessingStatusResponse

# Import existing functions (adapting path as needed)
import sys
sys.path.append('.')  # Ensure current directory is in path
from transcriber import get_youtube_transcript, transcribe_video
from downloader import download_video

router = APIRouter()

# Dictionary to track background tasks
task_status = {}

async def process_video_transcription(task_id: str, video_path: str, youtube_id: Optional[str] = None):
    """Background task to process video transcription"""
    try:
        task_status[task_id] = {"status": "processing", "progress": 0}
        
        # First check if it's a YouTube video with available transcript
        if youtube_id:
            task_status[task_id]["progress"] = 10
            transcript_segments, full_transcript, detected_language = await get_youtube_transcript(youtube_id)
            
            # If YouTube transcript is available, use it
            if transcript_segments:
                task_status[task_id]["progress"] = 100
                
                # Save transcript info
                transcript_file_path = os.path.join(settings.TRANSCRIPTS_DIR, f"{task_id}_transcript.txt")
                with open(transcript_file_path, 'w', encoding='utf-8') as f:
                    f.write(full_transcript)
                
                task_status[task_id] = {
                    "status": "completed", 
                    "progress": 100,
                    "result": {
                        "transcript_segments": transcript_segments,
                        "full_transcript": full_transcript,
                        "detected_language": detected_language,
                        "transcript_file_url": f"/downloads/transcripts/{os.path.basename(transcript_file_path)}"
                    }
                }
                return
        
        # If no YouTube transcript, transcribe with Whisper
        task_status[task_id]["progress"] = 20
        transcript_segments, full_transcript, detected_language = await transcribe_video(video_path)
        
        if transcript_segments:
            task_status[task_id]["progress"] = 95
            
            # Save transcript to file
            transcript_file_path = os.path.join(settings.TRANSCRIPTS_DIR, f"{task_id}_transcript.txt")
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                f.write(full_transcript)
            
            task_status[task_id] = {
                "status": "completed", 
                "progress": 100,
                "result": {
                    "transcript_segments": transcript_segments,
                    "full_transcript": full_transcript,
                    "detected_language": detected_language,
                    "transcript_file_url": f"/downloads/transcripts/{os.path.basename(transcript_file_path)}"
                }
            }
        else:
            task_status[task_id] = {
                "status": "failed", 
                "progress": 100,
                "error": "Failed to transcribe video"
            }
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }

@router.post("/transcribe/youtube", response_model=ProcessingStatusResponse, 
              summary="Transcribe a YouTube video")
async def transcribe_youtube_video(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks
):
    """
    Transcribe a YouTube video given its URL.
    
    - **youtube_url**: The URL of the YouTube video to transcribe
    
    Returns a task ID for checking the status of the transcription process.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Download the video
        downloaded_file, video_title, video_description, video_id = await download_video(request.youtube_url)
        
        if not downloaded_file or not video_id:
            raise HTTPException(status_code=400, detail="Failed to download YouTube video")
        
        # Start processing in background
        background_tasks.add_task(process_video_transcription, task_id, downloaded_file, video_id)
        
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

@router.post("/transcribe/upload", response_model=ProcessingStatusResponse,
             summary="Transcribe an uploaded video")
async def transcribe_uploaded_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Transcribe an uploaded video file.
    
    - **file**: The video file to transcribe
    
    Returns a task ID for checking the status of the transcription process.
    """
    task_id = str(uuid.uuid4())
    
    # Check file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB
    
    # Save the uploaded file to the temp directory
    file_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")
    
    try:
        with open(file_path, 'wb') as f:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > settings.MAX_UPLOAD_SIZE * 1024 * 1024:
                    # Clean up the partial file
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE}MB."
                    )
                f.write(chunk)
        
        # Start processing in background
        background_tasks.add_task(process_video_transcription, task_id, file_path)
        
        return ProcessingStatusResponse(
            task_id=task_id,
            status="processing",
            progress=0,
            result_url=None
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

@router.get("/transcribe/status/{task_id}", response_model=ProcessingStatusResponse, 
            summary="Check transcription status")
async def check_transcription_status(task_id: str):
    """
    Check the status of a transcription task.
    
    - **task_id**: The ID of the task to check
    
    Returns the current status of the transcription process.
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
        response.result_url = result.get("transcript_file_url")
    
    return response

@router.get("/transcribe/result/{task_id}", response_model=TranscriptionResponse,
           summary="Get transcription results")
async def get_transcription_result(task_id: str):
    """
    Get the results of a completed transcription task.
    
    - **task_id**: The ID of the completed task
    
    Returns the transcription results.
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
    
    return TranscriptionResponse(**status_info["result"])

@router.get("/transcripts/{filename}", summary="Download a transcript file")
async def get_transcript_file(filename: str):
    """
    Download a specific transcript file.
    
    - **filename**: The name of the transcript file to download
    
    Returns the transcript file.
    """
    file_path = os.path.join(settings.TRANSCRIPTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Transcript file {filename} not found")
    
    return FileResponse(
        file_path, 
        media_type="text/plain", 
        filename=filename
    )