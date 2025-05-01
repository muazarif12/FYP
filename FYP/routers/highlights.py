from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import asyncio
import uuid
from typing import Optional, List

from core.config import settings
from core.models import (
    HighlightsRequest, 
    HighlightsResponse, 
    ProcessingStatusResponse,
    HighlightSegment
)

# Import existing functions (adapting path as needed)
import sys
sys.path.append('.')  # Ensure current directory is in path
from highlights import generate_highlights, generate_custom_highlights
from algorithmic_highlights import generate_highlights_algorithmically
from downloader import download_video

router = APIRouter()

# Dictionary to track background tasks
task_status = {}

async def process_highlights_generation(
    task_id: str, 
    video_path: str, 
    transcript_segments: List, 
    video_info: dict,
    target_duration: Optional[int] = None,
    is_reel: bool = False,
    custom_instructions: Optional[str] = None,
    use_algorithmic: bool = False
):
    """Background task to process highlights generation"""
    try:
        task_status[task_id] = {"status": "processing", "progress": 0}
        
        # Update progress
        task_status[task_id]["progress"] = 30
        
        # Choose the appropriate highlight generation function
        if custom_instructions:
            # Generate custom highlights with specific instructions
            highlights_path, highlight_segments = await generate_custom_highlights(
                video_path,
                transcript_segments,
                video_info,
                custom_instructions,
                target_duration=target_duration
            )
        elif use_algorithmic:
            # Use faster algorithmic approach
            highlights_path, highlight_segments = await generate_highlights_algorithmically(
                video_path,
                transcript_segments,
                video_info,
                target_duration=target_duration,
                is_reel=is_reel
            )
        else:
            # Use LLM-based approach (original method)
            highlights_path, highlight_segments = await generate_highlights(
                video_path,
                transcript_segments,
                video_info,
                target_duration=target_duration,
                is_reel=is_reel
            )
        
        # Update progress
        task_status[task_id]["progress"] = 100
        
        if highlights_path and os.path.exists(highlights_path):
            # Calculate total duration
            total_duration = sum(segment["end"] - segment["start"] for segment in highlight_segments)
            
            # Move the highlights file to the highlights directory with the task ID
            filename = os.path.basename(highlights_path)
            new_path = os.path.join(settings.HIGHLIGHTS_DIR, f"{task_id}_{filename}")
            os.rename(highlights_path, new_path)
            
            # Create segments for the response
            segments = [
                HighlightSegment(
                    start=segment["start"],
                    end=segment["end"],
                    description=segment["description"]
                )
                for segment in highlight_segments
            ]
            
            # Set task as completed
            task_status[task_id] = {
                "status": "completed", 
                "progress": 100,
                "result": {
                    "highlights_url": f"/downloads/highlights/{os.path.basename(new_path)}",
                    "segments": segments,
                    "total_duration": total_duration
                }
            }
        else:
            task_status[task_id] = {
                "status": "failed", 
                "progress": 100,
                "error": "Failed to generate highlights"
            }
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }

@router.post("/highlights/generate", response_model=ProcessingStatusResponse, 
              summary="Generate video highlights")
async def generate_video_highlights(
    request: HighlightsRequest,
    background_tasks: BackgroundTasks,
    fast: bool = Query(False, description="Use faster algorithmic method instead of LLM")
):
    """
    Generate highlights from a YouTube video or uploaded video.
    
    - **video_id**: ID of the uploaded video (if already uploaded)
    - **youtube_url**: YouTube URL (will be downloaded)
    - **transcript_id**: ID of an existing transcript (if already processed)
    - **duration**: Target duration in seconds
    - **is_reel**: Generate a short social media reel
    - **custom_instructions**: Custom instructions for highlight generation
    - **fast**: Use faster algorithmic method instead of LLM-based approach
    
    Returns a task ID for checking the status of the highlights generation process.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Check that we have a video source
        if not request.video_id and not request.youtube_url:
            raise HTTPException(status_code=400, detail="Either video_id or youtube_url is required")
        
        # Get transcript segments
        transcript_segments = None
        if request.transcript_id:
            # Load transcript from file
            transcript_file = os.path.join(settings.TRANSCRIPTS_DIR, f"{request.transcript_id}_transcript.txt")
            
            if not os.path.exists(transcript_file):
                raise HTTPException(status_code=404, detail=f"Transcript {request.transcript_id} not found")
            
            # Logic to load transcript segments from file
            # This is a placeholder - you'll need to implement this based on your transcript format
            # transcript_segments = load_transcript_segments_from_file(transcript_file)
            
            # Placeholder for now - you'll need to get this from your transcription service
            raise HTTPException(status_code=400, detail="Loading from transcript_id not yet implemented")
        
        # Get the video file
        video_path = None
        video_info = {}
        
        if request.youtube_url:
            # Download from YouTube
            downloaded_file, video_title, video_description, video_id = await download_video(request.youtube_url)
            
            if not downloaded_file:
                raise HTTPException(status_code=400, detail="Failed to download YouTube video")
            
            video_path = downloaded_file
            video_info = {"title": video_title, "description": video_description}
            
            # If we don't have transcript segments already, get them from YouTube
            if not transcript_segments:
                from transcriber import get_youtube_transcript, transcribe_video
                
                # Try YouTube transcript first
                if video_id:
                    transcript_segments, _, _ = await get_youtube_transcript(video_id)
                
                # Fall back to Whisper transcription
                if not transcript_segments:
                    transcript_segments, _, _ = await transcribe_video(video_path)
                    
                if not transcript_segments:
                    raise HTTPException(status_code=400, detail="Failed to transcribe video")
        
        elif request.video_id:
            # Get video from storage
            video_path = os.path.join(settings.TEMP_DIR, f"{request.video_id}")
            
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")
            
            # If we don't have transcript segments, transcribe the video
            if not transcript_segments:
                from transcriber import transcribe_video
                transcript_segments, _, _ = await transcribe_video(video_path)
                
                if not transcript_segments:
                    raise HTTPException(status_code=400, detail="Failed to transcribe video")
        
        # Start processing in background
        background_tasks.add_task(
            process_highlights_generation,
            task_id,
            video_path,
            transcript_segments,
            video_info,
            request.duration,
            request.is_reel,
            request.custom_instructions,
            fast
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

@router.get("/highlights/status/{task_id}", response_model=ProcessingStatusResponse, 
            summary="Check highlights generation status")
async def check_highlights_status(task_id: str):
    """
    Check the status of a highlights generation task.
    
    - **task_id**: The ID of the task to check
    
    Returns the current status of the highlights generation process.
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
        response.result_url = result.get("highlights_url")
    
    return response

@router.get("/highlights/result/{task_id}", response_model=HighlightsResponse,
           summary="Get highlights generation results")
async def get_highlights_result(task_id: str):
    """
    Get the results of a completed highlights generation task.
    
    - **task_id**: The ID of the completed task
    
    Returns the highlights generation results.
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
    
    return HighlightsResponse(**status_info["result"])

@router.get("/highlights/{filename}", summary="Download a highlights video")
async def get_highlights_file(filename: str):
    """
    Download a specific highlights video file.
    
    - **filename**: The name of the highlights file to download
    
    Returns the highlights video file.
    """
    file_path = os.path.join(settings.HIGHLIGHTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Highlights file {filename} not found")
    
    return FileResponse(
        file_path, 
        media_type="video/mp4", 
        filename=filename
    )