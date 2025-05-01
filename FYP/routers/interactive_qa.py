from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid
import time

from core.config import settings

# Import from utils router to access task storage
from routers.utils import task_storage

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from video_qa import answer_video_question

router = APIRouter()

# Dictionary to track Q&A tasks
qa_task_storage = {}

# Data models
class VideoQARequest(BaseModel):
    task_id: str  # Task ID from video processing
    question: str
    generate_clip: bool = True  # Default to generating a clip

class VideoQAResponse(BaseModel):
    qa_task_id: str
    status: str = "processing"
    message: str

class VideoQAResult(BaseModel):
    answer: str
    timestamps: List[str]
    clip_url: Optional[str] = None
    clip_title: Optional[str] = None

@router.post("/interactive-qa", response_model=VideoQAResponse, tags=["Interactive Q&A"])
async def interactive_qa(request: VideoQARequest, background_tasks: BackgroundTasks):
    """
    Ask a question about the video and get an answer with relevant video clip and timestamps.
    
    - **task_id**: Task ID from video processing
    - **question**: Question about the video content
    - **generate_clip**: Whether to generate a video clip (default: True)
    
    Returns a task ID to track the Q&A processing.
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
    
    # Create a new task ID for this Q&A request
    qa_task_id = str(uuid.uuid4())
    
    try:
        # Start Q&A processing in background
        background_tasks.add_task(
            process_video_qa,
            qa_task_id,
            request.task_id,
            request.question,
            request.generate_clip
        )
        
        # Initialize task status
        qa_task_storage[qa_task_id] = {
            "status": "processing",
            "message": "Processing your question..."
        }
        
        return VideoQAResponse(
            qa_task_id=qa_task_id,
            status="processing",
            message="Your question is being processed. Use the qa_task_id to check status and get results."
        )
    
    except Exception as e:
        qa_task_storage[qa_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting Q&A processing: {str(e)}")

@router.get("/interactive-qa/status/{qa_task_id}", tags=["Interactive Q&A"])
async def check_qa_status(qa_task_id: str):
    """
    Check the status of an interactive Q&A task.
    
    - **qa_task_id**: Q&A task ID to check
    
    Returns the current status of the Q&A processing.
    """
    if qa_task_id not in qa_task_storage:
        raise HTTPException(status_code=404, detail=f"Q&A task {qa_task_id} not found")
    
    return qa_task_storage[qa_task_id]

@router.get("/interactive-qa/result/{qa_task_id}", response_model=VideoQAResult, tags=["Interactive Q&A"])
async def get_qa_result(qa_task_id: str):
    """
    Get the results of a completed interactive Q&A task.
    
    - **qa_task_id**: Q&A task ID
    
    Returns the answer, timestamps, and clip URL if available.
    """
    if qa_task_id not in qa_task_storage:
        raise HTTPException(status_code=404, detail=f"Q&A task {qa_task_id} not found")
    
    task_info = qa_task_storage[qa_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Q&A task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info:
        raise HTTPException(status_code=500, detail="No result available")
    
    return VideoQAResult(**task_info["result"])

@router.get("/interactive-qa/clip/{qa_task_id}", tags=["Interactive Q&A"])
async def get_qa_clip(qa_task_id: str):
    """
    Download the video clip for a Q&A response.
    
    - **qa_task_id**: Q&A task ID
    
    Returns the video clip file.
    """
    if qa_task_id not in qa_task_storage:
        raise HTTPException(status_code=404, detail=f"Q&A task {qa_task_id} not found")
    
    task_info = qa_task_storage[qa_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Q&A task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "clip_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No clip available for this Q&A")
    
    clip_path = task_info["result"]["clip_path"]
    
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Clip file not found")
    
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=f"answer_{qa_task_id}.mp4"
    )

# Background task for processing Q&A
async def process_video_qa(qa_task_id: str, video_task_id: str, question: str, generate_clip: bool):
    """Background task to process video Q&A with clip generation"""
    try:
        qa_task_storage[qa_task_id] = {
            "status": "processing",
            "message": "Processing your question..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        video_path = task_info["video_info"]["path"]
        transcript_segments = task_info["transcript_info"]["segments"]
        full_text = task_info["transcript_info"]["full_text"]
        
        # Create output directory for clips if it doesn't exist
        qa_clips_dir = os.path.join(settings.OUTPUT_DIR, "qa_clips")
        os.makedirs(qa_clips_dir, exist_ok=True)
        
        # Process Q&A
        qa_result = await answer_video_question(
            transcript_segments,
            video_path,
            question,
            full_text=full_text,
            generate_clip=generate_clip
        )
        
        # If clip was generated, make it accessible via API
        clip_url = None
        clip_path = None
        
        if qa_result.get("clip_path") and os.path.exists(qa_result["clip_path"]):
            # Rename and move the clip to a more permanent location
            original_clip = qa_result["clip_path"]
            clip_filename = f"{qa_task_id}_answer.mp4"
            clip_path = os.path.join(qa_clips_dir, clip_filename)
            
            # Move the file
            os.rename(original_clip, clip_path)
            
            # Create accessible URL
            clip_url = f"/api/interactive-qa/clip/{qa_task_id}"
        
        # Store the result
        qa_task_storage[qa_task_id] = {
            "status": "completed",
            "result": {
                "answer": qa_result.get("answer", ""),
                "timestamps": qa_result.get("formatted_timestamps", []),
                "clip_url": clip_url,
                "clip_title": qa_result.get("clip_title", ""),
                "clip_path": clip_path  # Store the actual path for internal use
            }
        }
        
    except Exception as e:
        qa_task_storage[qa_task_id] = {
            "status": "failed",
            "error": str(e)
        }