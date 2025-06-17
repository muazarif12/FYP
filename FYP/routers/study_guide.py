from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid
import shutil

from study_guide import generate_suggested_questions
from core.config import settings

# Import from utils router to access task storage
from routers.utils import task_storage

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from study_guide import generate_study_guide, generate_faq

router = APIRouter()

# Dictionary to track study guide tasks
study_guide_task_storage = {}

# Data models
class StudyGuideRequest(BaseModel):
    task_id: str  # Task ID from video processing

class StudyGuideResponse(BaseModel):
    study_guide_task_id: str
    status: str = "processing"
    message: str


class SuggestedQuestionsRequest(BaseModel):  # Renamed from FaqRequest
    task_id: str  # Task ID from video processing

class SuggestedQuestionsResult(BaseModel):  # Renamed from FaqResult
    questions: List[str]  # Changed to list of strings instead of dict

class FaqRequest(BaseModel):
    task_id: str  # Task ID from video processing

class FaqResult(BaseModel):
    questions: List[Dict[str, str]]




@router.post("/suggested-questions", response_model=SuggestedQuestionsResult, tags=["Chatbot"])  # Updated endpoint
async def generate_suggested_questions_for_chatbot(request: SuggestedQuestionsRequest):
    """
    Generate suggested questions about the video content for a chatbot interface.
    
    - **task_id**: Task ID from video processing
    
    Returns a list of suggested questions that users might want to ask about the video.
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
    
    try:
        # Get transcript info
        transcript_segments = task_info["transcript_info"]["segments"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Generate suggested questions
        suggested_questions = await generate_suggested_questions(transcript_segments, video_info)
        
        return SuggestedQuestionsResult(questions=suggested_questions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggested questions: {str(e)}")



@router.post("/study-guide", response_model=StudyGuideResponse, tags=["Study Guide"])
async def create_study_guide(request: StudyGuideRequest, background_tasks: BackgroundTasks):
    """
    Generate a comprehensive study guide from the video content.
    
    - **task_id**: Task ID from video processing
    
    Returns a task ID to track the study guide generation.
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
    
    # Create a new task ID for this study guide request
    study_guide_task_id = str(uuid.uuid4())
    
    try:
        # Start study guide generation in background
        background_tasks.add_task(
            process_study_guide_generation,
            study_guide_task_id,
            request.task_id
        )
        
        # Initialize task status
        study_guide_task_storage[study_guide_task_id] = {
            "status": "processing",
            "message": "Generating study guide..."
        }
        
        return StudyGuideResponse(
            study_guide_task_id=study_guide_task_id,
            status="processing",
            message="Your study guide is being generated. Use the study_guide_task_id to check status and get results."
        )
    
    except Exception as e:
        study_guide_task_storage[study_guide_task_id] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error starting study guide generation: {str(e)}")

@router.get("/study-guide/status/{study_guide_task_id}", tags=["Study Guide"])
async def check_study_guide_status(study_guide_task_id: str):
    """
    Check the status of a study guide generation task.
    
    - **study_guide_task_id**: Study guide task ID to check
    
    Returns the current status of the study guide generation.
    """
    if study_guide_task_id not in study_guide_task_storage:
        raise HTTPException(status_code=404, detail=f"Study guide task {study_guide_task_id} not found")
    
    return study_guide_task_storage[study_guide_task_id]

@router.get("/study-guide/download/{study_guide_task_id}", tags=["Study Guide"])
async def download_study_guide(study_guide_task_id: str):
    """
    Download the generated study guide.
    
    - **study_guide_task_id**: Study guide task ID
    
    Returns the study guide file (markdown).
    """
    if study_guide_task_id not in study_guide_task_storage:
        raise HTTPException(status_code=404, detail=f"Study guide task {study_guide_task_id} not found")
    
    task_info = study_guide_task_storage[study_guide_task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Study guide task not complete. Current status: {task_info['status']}"
        )
    
    if "result" not in task_info or "file_path" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="No study guide available for download")
    
    file_path = task_info["result"]["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Study guide file not found")
    
    return FileResponse(
        file_path,
        media_type="text/markdown",
        filename=f"study_guide_{study_guide_task_id}.md"
    )

@router.post("/faq", response_model=FaqResult, tags=["Study Guide"])
async def generate_video_faq(request: FaqRequest):
    """
    Generate frequently asked questions (FAQ) about the video content.
    
    - **task_id**: Task ID from video processing
    
    Returns a list of questions and answers about the video.
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
    
    try:
        # Get transcript info
        transcript_segments = task_info["transcript_info"]["segments"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Generate FAQ
        faq_questions = await generate_faq(transcript_segments, video_info)
        
        return FaqResult(questions=faq_questions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating FAQ: {str(e)}")

# Background task for processing study guide generation
async def process_study_guide_generation(study_guide_task_id: str, video_task_id: str):
    """Background task to generate study guide from video"""
    try:
        study_guide_task_storage[study_guide_task_id] = {
            "status": "processing",
            "message": "Generating study guide..."
        }
        
        # Get video and transcript info from the original task
        task_info = task_storage[video_task_id]
        transcript_segments = task_info["transcript_info"]["segments"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Create output directory for study guides if it doesn't exist
        study_guides_dir = os.path.join(settings.OUTPUT_DIR, "study_guides")
        os.makedirs(study_guides_dir, exist_ok=True)
        
        # Generate study guide
        study_guide_result = await generate_study_guide(transcript_segments, video_info)
        
        if not study_guide_result or "file_path" not in study_guide_result:
            study_guide_task_storage[study_guide_task_id] = {
                "status": "failed",
                "error": "Failed to generate study guide"
            }
            return
        
        # Rename and move file to a more permanent location
        original_path = study_guide_result["file_path"]
        new_filename = f"{study_guide_task_id}_study_guide.md"
        new_path = os.path.join(study_guides_dir, new_filename)
        
        # Copy the file
        shutil.copy(original_path, new_path)
        
        # Store result info
        study_guide_task_storage[study_guide_task_id] = {
            "status": "completed",
            "result": {
                "title": study_guide_result.get("study_guide", {}).get("title", "Study Guide"),
                "file_path": new_path,
                "download_url": f"/api/study-guide/download/{study_guide_task_id}"
            }
        }
        
    except Exception as e:
        import traceback
        study_guide_task_storage[study_guide_task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }