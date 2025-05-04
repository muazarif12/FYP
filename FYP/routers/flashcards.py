from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import os
import uuid
import sys

# bring in your existing task storage and settings
from routers.utils import task_storage
from core.config import settings

# ensure our flashcards generator is importable
sys.path.append('.')
from flash_cards import generate_flashcards  # your async function from above

router = APIRouter()
flashcard_task_storage: Dict[str, Dict] = {}

# --- Pydantic models ----------------------------------

class FlashcardsRequest(BaseModel):
    task_id: str  # Task ID from video processing

class FlashcardsResponse(BaseModel):
    flashcards_task_id: str
    status: str = "processing"
    message: str

class FlashcardResult(BaseModel):
    cards: List[Dict[str, str]]  # each {"front": str, "back": str}

# --- Endpoints ----------------------------------------

@router.post("/flashcards", response_model=FlashcardsResponse, tags=["Flashcards"])
async def create_flashcards(request: FlashcardsRequest, background_tasks: BackgroundTasks):
    """
    Generate flashcards from the video content.

    - **task_id**: Task ID from video processing
    """
    # 1) verify original video task exists and completed
    if request.task_id not in task_storage:
        raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")

    orig = task_storage[request.task_id]
    if orig["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video processing not complete. Current status: {orig['status']}"
        )

    # 2) create our flashcards task
    flash_id = str(uuid.uuid4())
    flashcard_task_storage[flash_id] = {
        "status": "processing",
        "message": "Generating flashcards..."
    }

    # 3) schedule background generation
    background_tasks.add_task(
        process_flashcards_generation,
        flash_id,
        request.task_id
    )

    return FlashcardsResponse(
        flashcards_task_id=flash_id,
        status="processing",
        message="Flashcards generation started; check status with this ID."
    )

@router.get("/flashcards/status/{flashcards_task_id}", tags=["Flashcards"])
async def check_flashcards_status(flashcards_task_id: str):
    """
    Check the status of a flashcards generation task.
    """
    if flashcards_task_id not in flashcard_task_storage:
        raise HTTPException(status_code=404, detail="Flashcards task not found")

    return flashcard_task_storage[flashcards_task_id]

@router.get("/flashcards/{flashcards_task_id}", response_model=FlashcardResult, tags=["Flashcards"])
async def get_flashcards(flashcards_task_id: str):
    """
    Retrieve generated flashcards (once complete).
    """
    if flashcards_task_id not in flashcard_task_storage:
        raise HTTPException(status_code=404, detail="Flashcards task not found")

    info = flashcard_task_storage[flashcards_task_id]
    if info["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Flashcards not ready. Current status: {info['status']}"
        )

    return FlashcardResult(cards=info["result"]["cards"])

# --- Background task handler -------------------------

async def process_flashcards_generation(flash_id: str, video_task_id: str):
    """
    Background task to call generate_flashcards() and store the result.
    """
    try:
        # pull transcript and video metadata
        orig = task_storage[video_task_id]
        segments = orig["transcript_info"]["segments"]
        video_info = {
            "title": orig["video_info"]["title"],
            "description": orig["video_info"]["description"]
        }

        # run your LLM-based generator
        cards = await generate_flashcards(segments, video_info)

        # store the completed result
        flashcard_task_storage[flash_id] = {
            "status": "completed",
            "result": {
                "cards": cards
            }
        }

    except Exception as e:
        import traceback
        flashcard_task_storage[flash_id] = {
            "status": "failed",
            "error": str(e),
            "trace": traceback.format_exc()
        }
