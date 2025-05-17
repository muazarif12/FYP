from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import asyncio
import uuid
import re

# Import from utils router to access task storage
from routers.utils import task_storage

# Import existing functions
import sys
sys.path.append('.')  # Ensure current directory is in path
from summarizer import generate_enhanced_response, generate_key_moments_algorithmically
from retrieval import retrieve_chunks, initialize_indexes
from highlights import extract_highlights, generate_highlights, generate_custom_highlights, merge_clips
from meeting_minutes import generate_meeting_minutes, save_meeting_minutes
from podcast_integration import generate_podcast
from dubbing import create_english_dub
from subtitling import create_english_subtitles
from study_guide import generate_study_guide, generate_faq
from video_qa import answer_video_question
from utils import format_timestamp

router = APIRouter()

# Data models
class ChatRequest(BaseModel):
    task_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    response_type: str
    extra_data: Optional[dict] = None

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_video(request: ChatRequest):
    """
    Chat with the video content. This endpoint handles all types of queries including:
    - General questions
    - Summary requests
    - Key moments/timeline requests
    - Highlight requests
    - Podcast requests
    - Study guide requests
    - Subtitles requests
    - Dubbing requests
    - Meeting minutes requests
    - Interactive Q&A requests
    
    The API will automatically detect the query type and respond accordingly.
    
    - **task_id**: Task ID from video processing
    - **message**: User's message or command
    
    Returns a response based on the video content and detected query type.
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
        # Extract transcript and video info
        transcript_segments = task_info["transcript_info"]["segments"]
        full_text = task_info["transcript_info"]["full_text"]
        detected_language = task_info["transcript_info"]["language"]
        video_path = task_info["video_info"]["path"]
        video_info = {
            "title": task_info["video_info"]["title"],
            "description": task_info["video_info"]["description"]
        }
        
        # Initialize retrieval indexes
        await initialize_indexes(full_text)
        
        # Extract user message
        user_input = request.message
        
        # Initialize variables
        target_duration = None
        custom_podcast_prompt = None
        
        # Identify query type - similar to your chatbot.py logic
        query_type = "general"
        
        # Check for podcast-related requests
        if re.search(r'podcast|conversation|discussion|dialogue|talk show|interview|audio', user_input.lower()):
            # Extract any specific instructions for podcast style
            style_match = re.search(r'(casual|funny|serious|educational|debate|friendly|professional|entertaining)', user_input.lower())
            format_match = re.search(r'style[:]?\s*(\w+)', user_input.lower())
            
            # Determine podcast style from the request
            if style_match:
                podcast_style = style_match.group(1)
                custom_podcast_prompt = f"Make the podcast {podcast_style} in tone and style."
                
            if format_match:
                podcast_format = format_match.group(1)
                if custom_podcast_prompt:
                    custom_podcast_prompt += f" Follow a {podcast_format} format."
                else:
                    custom_podcast_prompt = f"Follow a {podcast_format} format."
            
            query_type = "podcast"
            
            # Return a message about podcast generation with instructions to use the dedicated endpoint
            return ChatResponse(
                response="To generate a podcast, please use the dedicated podcast endpoint: POST /api/podcast with your task_id and optional style parameter.",
                response_type="podcast_info",
                extra_data={
                    "instruction": "Use the podcast endpoint",
                    "style": custom_podcast_prompt
                }
            )
            
        elif re.search(r'english subtitles|add subtitles|create subtitles|subtitle', user_input.lower()):
            query_type = "english_subtitles"
            return ChatResponse(
                response="To generate English subtitles, please use the dedicated subtitles endpoint: POST /api/subtitles with your task_id.",
                response_type="subtitles_info"
            )
            
        elif re.search(r'study guide|study material|learning guide|course notes', user_input.lower()):
            query_type = "study_guide"
            return ChatResponse(
                response="To generate a study guide, please use the dedicated study guide endpoint: POST /api/study-guide with your task_id.",
                response_type="study_guide_info"
            )
            
        elif re.search(r'meeting minutes|minutes|meeting notes|meeting summary', user_input.lower()):
            query_type = "meeting_minutes"
            # For meeting minutes, we can actually generate a summary response here
            # This gives a preview of what the full meeting minutes would contain
            retrieved_docs = await retrieve_chunks(full_text, "meeting summary key points", k=5)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            meeting_summary = await generate_enhanced_response(
                "summary", 
                references, 
                "Create a brief summary of this meeting focusing on key decisions and action items", 
                detected_language
            )
            
            return ChatResponse(
                response=f"Meeting Minutes Preview:\n\n{meeting_summary}\n\nTo generate complete meeting minutes, please use the dedicated endpoint: POST /api/meeting-minutes with your task_id.",
                response_type="meeting_minutes_preview"
            )
            
        elif re.search(r'english dub|dub|dubbing|voice over|translate audio|translate voice', user_input.lower()):
            query_type = "english_dub"
            
            # Check if dubbing is applicable (non-English video)
            if detected_language == "en":
                return ChatResponse(
                    response="The video is already in English, so dubbing is not necessary.",
                    response_type="dubbing_info"
                )
            else:
                return ChatResponse(
                    response=f"To generate English dubbing for this {detected_language} video, please use the dedicated dubbing endpoint: POST /api/dub with your task_id.",
                    response_type="dubbing_info"
                )
            
        elif re.search(r'^(extract\s+clips?|extracting\s+clips?|interactive(\s+q/?a)?|video\s+q/?a|create\s+clips?|find\s+clips?|clip\s+extract|get\s+clips?|show\s+clips?|video\s+answer|answer\s+with\s+clip)s?:?', user_input, re.IGNORECASE):
            query_type = "interactive_qa"
            # Extract the actual question by removing the trigger phrase
            actual_question = re.sub(r'^(extract\s+clips?|extracting\s+clips?|interactive(\s+q/?a)?|video\s+q/?a|create\s+clips?|find\s+clips?|clip\s+extract|get\s+clips?|show\s+clips?|video\s+answer|answer\s+with\s+clip)s?:?\s*', '', user_input, flags=re.IGNORECASE)
            
            return ChatResponse(
                response=f"To get an answer with video clips for your question: '{actual_question}', please use the dedicated interactive Q&A endpoint: POST /api/interactive-qa with your task_id and question.",
                response_type="interactive_qa_info",
                extra_data={
                    "question": actual_question
                }
            )
            
        # Check for highlight-related requests since they can overlap with other patterns
        elif re.search(r'highlight|best parts|important parts', user_input.lower()):
            # Extract duration information if explicitly specified
            duration_match = re.search(r'(\d+)\s*(minute|min|minutes|second|sec|seconds)', user_input.lower())

            if duration_match:
                amount = int(duration_match.group(1))
                unit = duration_match.group(2)
                if unit.startswith('minute') or unit.startswith('min'):
                    target_duration = amount * 60
                else:
                    target_duration = amount
                query_type = "custom_duration_highlights"
            # Check for custom instructions in the request
            elif re.search(r'(where|ensure|keep|include|focus on|select|show|add|include|take|at timestamp|first moment|last moment|beginning|end|intro|conclusion)', user_input.lower()):
                query_type = "custom_prompt_highlights"
            else:
                query_type = "highlights"
                
            # Check if user requested fast generation explicitly
            use_algorithmic = "fast" in user_input.lower() or "quick" in user_input.lower() or "algorithmic" in user_input.lower()
                
            return ChatResponse(
                response=f"To generate video highlights, please use the dedicated highlights endpoint: POST /api/highlights with your task_id, duration_seconds: {target_duration if target_duration else 'null'}, custom_prompt: {user_input if query_type == 'custom_prompt_highlights' else 'null'}, and use_algorithmic: {str(use_algorithmic).lower()}.",
                response_type="highlights_info",
                extra_data={
                    "duration_seconds": target_duration,
                    "custom_prompt": user_input if query_type == "custom_prompt_highlights" else None,
                    "use_algorithmic": use_algorithmic
                }
            )
            
        elif any(term in user_input.lower() for term in ["timeline", "key moments", "chapters", "sections", "timestamps"]):
            query_type = "key_moments"
            
            # For key moments, we can generate them here
            try:
                # Get full transcript text
                full_timestamped_transcript = full_text  # This might need adjustment based on your full_text format
                
                # Generate key moments
                key_moments_structured, key_moments_formatted = await generate_key_moments_algorithmically(
                    transcript_segments, 
                    full_timestamped_transcript,
                    detected_language
                )
                
                # Create a response in the detected language
                if detected_language == "en":
                    response_prefix = "Here are the key moments in the video:\n\n"
                else:
                    # This would need to be translated in a production implementation
                    response_prefix = "Here are the key moments in the video:\n\n"
                
                return ChatResponse(
                    response=f"{response_prefix}{key_moments_formatted}",
                    response_type="key_moments",
                    extra_data={
                        "key_moments": key_moments_structured
                    }
                )
            except Exception as e:
                # If there's an error generating key moments, return a generic response
                return ChatResponse(
                    response=f"I encountered an error generating key moments: {str(e)}",
                    response_type="error"
                )
                
        elif any(term in user_input.lower() for term in ["summarize", "summary", "overview", "what is the video about"]):
            query_type = "summary"
            
            # For summary, we'll use the existing logic but return the result directly
            retrieved_docs = await retrieve_chunks(full_text, user_input, k=5)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            summary = await generate_enhanced_response(
                query_type, 
                references, 
                user_input, 
                detected_language
            )
            
            return ChatResponse(
                response=summary,
                response_type="summary"
            )
            
        elif any(term in user_input.lower() for term in ["key topics", "main points", "main ideas", "central themes"]):
            query_type = "key_topics"
            
            # For key topics, use the enhanced response function
            retrieved_docs = await retrieve_chunks(full_text, user_input, k=5)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            key_topics = await generate_enhanced_response(
                query_type, 
                references, 
                user_input, 
                detected_language
            )
            
            return ChatResponse(
                response=key_topics,
                response_type="key_topics"
            )
            
        elif re.search(r'at (\d{1,2}:?\d{1,2}:?\d{0,2})|timestamp|(\d{1,2}:\d{2})', user_input.lower()):
            query_type = "specific_timestamp"
            
            # For specific timestamp queries, use the enhanced response function
            retrieved_docs = await retrieve_chunks(full_text, user_input, k=3)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            timestamp_info = await generate_enhanced_response(
                query_type, 
                references, 
                user_input, 
                detected_language
            )
            
            return ChatResponse(
                response=timestamp_info,
                response_type="specific_timestamp"
            )
            
        elif re.search(r'reel|short clip|tiktok|instagram|social media', user_input.lower()):
            query_type = "reel"
            
            # Check if user requested fast generation
            use_algorithmic = "fast" in user_input.lower() or "quick" in user_input.lower() or "algorithmic" in user_input.lower()
            
            return ChatResponse(
                response=f"To generate a short reel for social media, please use the dedicated highlights endpoint: POST /api/highlights with your task_id, duration_seconds: 60, is_reel: true, and use_algorithmic: {str(use_algorithmic).lower()}.",
                response_type="reel_info",
                extra_data={
                    "duration_seconds": 60,
                    "is_reel": True,
                    "use_algorithmic": use_algorithmic
                }
            )
            
        else:
            # For general questions, use the standard retrieval and generation approach
            retrieved_docs = await retrieve_chunks(full_text, user_input, k=3)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            response = await generate_enhanced_response(
                "general", 
                references, 
                user_input, 
                detected_language
            )
            
            return ChatResponse(
                response=response,
                response_type="general"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Additional endpoints for specific features will be handled by their respective routers