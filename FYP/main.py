from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import asyncio
import shutil
import uuid
import uvicorn
from typing import Optional, List

# Import routers
from routers import transcription, highlights, podcast, chat, utils

# Create FastAPI app
app = FastAPI(
    title="VidSense API",
    description="API for video analysis and content generation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files directory for serving generated content
app.mount("/downloads", StaticFiles(directory=OUTPUT_DIR), name="downloads")

# Include routers
app.include_router(transcription.router, prefix="/api", tags=["Transcription"])
app.include_router(highlights.router, prefix="/api", tags=["Highlights"])
app.include_router(podcast.router, prefix="/api", tags=["Podcast"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(utils.router, prefix="/api", tags=["Utilities"])

@app.get("/")
async def root():
    return {"message": "Welcome to VidSense API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)