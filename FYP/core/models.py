from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
import re

class YouTubeRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL")
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        if not re.match(r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]+', v):
            raise ValueError("Invalid YouTube URL format")
        return v

class TranscriptionResponse(BaseModel):
    transcript_segments: List[List[Union[float, str]]] = Field(..., description="List of transcript segments [start_time, end_time, text]")
    full_transcript: str = Field(..., description="Full transcript text")
    detected_language: str = Field(..., description="Detected language code")
    transcript_file_url: Optional[str] = Field(None, description="URL to download the transcript file")

class SummaryRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="Video ID if already uploaded/processed")
    youtube_url: Optional[str] = Field(None, description="YouTube URL to process")
    transcript_id: Optional[str] = Field(None, description="Transcript ID if already processed")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Video summary")
    key_points: List[str] = Field(..., description="Key points from the video")
    topics: List[Dict[str, str]] = Field(..., description="Key topics covered")

class HighlightsRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="Video ID if already uploaded/processed")
    youtube_url: Optional[str] = Field(None, description="YouTube URL to process")
    transcript_id: Optional[str] = Field(None, description="Transcript ID if already processed")
    duration: Optional[int] = Field(None, description="Target duration in seconds")
    is_reel: bool = Field(False, description="Generate a short reel instead of highlights")
    custom_instructions: Optional[str] = Field(None, description="Custom instructions for highlights generation")

class HighlightSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    description: str = Field(..., description="Description of the segment")

class HighlightsResponse(BaseModel):
    highlights_url: str = Field(..., description="URL to the generated highlights video")
    segments: List[HighlightSegment] = Field(..., description="Highlight segments")
    total_duration: float = Field(..., description="Total duration of highlights in seconds")

class PodcastRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="Video ID if already uploaded/processed")
    youtube_url: Optional[str] = Field(None, description="YouTube URL to process")
    transcript_id: Optional[str] = Field(None, description="Transcript ID if already processed")
    style: Optional[str] = Field(None, description="Podcast style (casual, formal, educational, etc.)")

class PodcastResponse(BaseModel):
    podcast_url: str = Field(..., description="URL to the generated podcast audio")
    transcript_url: str = Field(..., description="URL to the podcast transcript")
    title: str = Field(..., description="Podcast title")
    duration: float = Field(..., description="Podcast duration in seconds")
    hosts: List[str] = Field(..., description="Podcast hosts")

class ChatRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="Video ID if already uploaded/processed")
    transcript_id: Optional[str] = Field(None, description="Transcript ID if already processed")
    message: str = Field(..., description="User's message/question")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    video_clip_url: Optional[str] = Field(None, description="URL to a related video clip if available")
    timestamps: List[str] = Field(default=[], description="Relevant timestamps in the video")

class VideoQARequest(BaseModel):
    video_id: str = Field(..., description="Video ID")
    question: str = Field(..., description="Question about the video")
    generate_clip: bool = Field(True, description="Whether to generate a video clip")

class VideoQAResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question")
    clip_path: Optional[str] = Field(None, description="Path to generated clip")
    formatted_timestamps: List[str] = Field(default=[], description="Formatted timestamps")
    clip_title: Optional[str] = Field(None, description="Title for the clip")
    processing_time: float = Field(..., description="Processing time in seconds")

class ProcessingStatusResponse(BaseModel):
    task_id: str = Field(..., description="Task ID for the processing job")
    status: str = Field(..., description="Status of the processing job")
    progress: float = Field(..., description="Progress percentage (0-100)")
    result_url: Optional[str] = Field(None, description="URL to the result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")