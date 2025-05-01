from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import asyncio
import uuid
from typing import Optional, List, Dict, Any

from core.config import settings
from core.models import (
    ChatRequest,
    ChatResponse,
    VideoQARequest,
    VideoQAResponse,
    ProcessingStatusResponse
)

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from summarizer import generate_enhanced_response
from retrieval import retrieve_chunks, initialize_indexes
from video_qa import answer_video_question
from downloader import download_video

router = APIRouter()

# Dictionary to track background tasks
task_status = {}

# Cache for processed transcripts to avoid re-processing
transcript_cache = {}

@router.post("/chat", response_model=ChatResponse, summary="Chat with the video content")
async def chat_with_video(request: ChatRequest):
    """
    Chat with the video content. Ask questions or request information about the video.
    
    - **video_id**: ID of the uploaded/processed video
    - **transcript_id**: ID of the processed transcript
    - **message**: User's message/question
    
    Returns a response based on the video content.
    """
    try:
        # Validate request
        if not request.video_id and not request.transcript_id:
            raise HTTPException(status_code=400, detail="Either video_id or transcript_id is required")
        
        # Get transcript data
        transcript_segments = None
        full_text = None
        
        if request.transcript_id:
            # Check if transcript is in cache
            if request.transcript_id in transcript_cache:
                transcript_segments = transcript_cache[request.transcript_id]["segments"]
                full_text = transcript_cache[request.transcript_id]["text"]
            else:
                # Load transcript from file
                transcript_file = os.path.join(settings.TRANSCRIPTS_DIR, f"{request.transcript_id}_transcript.txt")
                
                if not os.path.exists(transcript_file):
                    raise HTTPException(status_code=404, detail=f"Transcript {request.transcript_id} not found")
                
                # Load the transcript segments and full text
                # This is a placeholder - you'll need to implement based on your format
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                # Parse transcript segments - implement this according to your format
                # This is just a placeholder
                transcript_segments = []  # parse_transcript_file(transcript_file)
                
                # Cache for future use
                transcript_cache[request.transcript_id] = {
                    "segments": transcript_segments,
                    "text": full_text
                }
        
        # If still no transcript, try to get from video file
        if not transcript_segments and request.video_id:
            video_path = os.path.join(settings.TEMP_DIR, f"{request.video_id}")
            
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")
            
            from transcriber import transcribe_video
            transcript_segments, full_text, _ = await transcribe_video(video_path)
            
            if not transcript_segments:
                raise HTTPException(status_code=400, detail="Failed to transcribe video")
        
        # If we only have segments and no full text, create full text
        if transcript_segments and not full_text:
            full_text = " ".join([seg[2] for seg in transcript_segments])
        
        # Initialize retrieval indexes if needed
        await initialize_indexes(full_text)
        
        # Determine query type
        query_type = "general"
        
        if any(term in request.message.lower() for term in ["timeline", "key moments", "chapters", "sections", "timestamps"]):
            query_type = "key_moments"
        elif any(term in request.message.lower() for term in ["summarize", "summary", "overview", "what is the video about"]):
            query_type = "summary"
        elif any(term in request.message.lower() for term in ["key topics", "main points", "main ideas", "central themes"]):
            query_type = "key_topics"
        
        # Retrieve relevant chunks
        retrieved_docs = await retrieve_chunks(full_text, request.message, k=3)
        references = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        response_text = await generate_enhanced_response(
            query_type, 
            references, 
            request.message, 
            detected_language="en"
        )
        
        # Format response
        return ChatResponse(
            response=response_text,
            video_clip_url=None,  # No clips for basic chat
            timestamps=[]  # No timestamps for basic chat
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video_qa", response_model=ProcessingStatusResponse, summary="Q&A with video clip generation")
async def video_qa_with_clip(
    request: VideoQARequest,
    background_tasks: BackgroundTasks
):
    """
    Ask a question about the video and get an answer with relevant video clips.
    
    - **video_id**: ID of the video to query
    - **question**: Question about the video content
    - **generate_clip**: Whether to generate a video clip
    
    Returns a task ID for checking the status of the video Q&A process.
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Validate request
        if not request.video_id:
            raise HTTPException(status_code=400, detail="video_id is required")
        
        # Get video path
        video_path = os.path.join(settings.TEMP_DIR, f"{request.video_id}")
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")
        
        # Start background task
        background_tasks.add_task(
            process_video_qa,
            task_id,
            video_path,
            request.video_id,
            request.question,
            request.generate_clip
        )
        
        # Return task ID
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

async def process_video_qa(
    task_id: str,
    video_path: str,
    video_id: str,
    question: str,
    generate_clip: bool
):
    """Background task to process video Q&A with clip generation"""
    try:
        task_status[task_id] = {"status": "processing", "progress": 0}
        
        # Get transcript
        transcript_segments = None
        full_text = None
        
        # Check if transcript is in cache
        if video_id in transcript_cache:
            transcript_segments = transcript_cache[video_id]["segments"]
            full_text = transcript_cache[video_id]["text"]
        else:
            # Transcribe the video
            from transcriber import transcribe_video
            transcript_segments, full_transcript, _ = await transcribe_video(video_path)
            
            if not transcript_segments:
                task_status[task_id] = {
                    "status": "failed", 
                    "progress": 100,
                    "error": "Failed to transcribe video"
                }
                return
            
            full_text = " ".join([seg[2] for seg in transcript_segments])
            
            # Cache for future use
            transcript_cache[video_id] = {
                "segments": transcript_segments,
                "text": full_text
            }
        
        # Update progress
        task_status[task_id]["progress"] = 30
        
        # Process Q&A with clip generation
        qa_result = await answer_video_question(
            transcript_segments,
            video_path,
            question,
            full_text=full_text,
            generate_clip=generate_clip
        )
        
        # Update progress
        task_status[task_id]["progress"] = 100
        
        # Format the clip path to be a URL if it exists
        clip_url = None
        if qa_result.get("clip_path"):
            clip_filename = os.path.basename(qa_result["clip_path"])
            
            # Move the clip to a more permanent location
            new_clip_path = os.path.join(settings.HIGHLIGHTS_DIR, f"{task_id}_{clip_filename}")
            os.rename(qa_result["clip_path"], new_clip_path)
            
            clip_url = f"/downloads/highlights/{os.path.basename(new_clip_path)}"
        
        # Set task as completed
        task_status[task_id] = {
            "status": "completed", 
            "progress": 100,
            "result": {
                "answer": qa_result.get("answer", ""),
                "clip_path": clip_url,
                "formatted_timestamps": qa_result.get("formatted_timestamps", []),
                "clip_title": qa_result.get("clip_title", ""),
                "processing_time": qa_result.get("processing_time", 0)
            }
        }
    
    except Exception as e:
        task_status[task_id] = {
            "status": "failed", 
            "progress": 100,
            "error": str(e)
        }

@router.get("/video_qa/status/{task_id}", response_model=ProcessingStatusResponse, 
            summary="Check video Q&A status")
async def check_video_qa_status(task_id: str):
    """
    Check the status of a video Q&A task.
    
    - **task_id**: The ID of the task to check
    
    Returns the current status of the video Q&A process.
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
        response.result_url = result.get("clip_path")
    
    return response

@router.get("/video_qa/result/{task_id}", response_model=VideoQAResponse,
           summary="Get video Q&A results")
async def get_video_qa_result(task_id: str):
    """
    Get the results of a completed video Q&A task.
    
    - **task_id**: The ID of the completed task
    
    Returns the video Q&A results including answer and clip if generated.
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
    
    return VideoQAResponse(**status_info["result"])