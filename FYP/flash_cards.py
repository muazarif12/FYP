import json
import re
import os
import asyncio
from logger_config import logger

# Your LLM client
from summarizer import generate_response_with_gemini_async
from transcriber import get_youtube_transcript, transcribe_video

async def generate_flashcards(transcript_segments, video_info):
    """
    Generate flashcards based on video transcript.

    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        video_info: Dict with 'title' and 'description'

    Returns:
        List[Dict[str, str]]: [{"front": "...", "back": "..."}, ...]
    """
    # 1) Combine transcript
    full_text = " ".join(seg[2] for seg in transcript_segments)

    # 2) Prompt the LLM without specifying a number
    prompt = f"""
You are an expert educator. Create flashcards for this video transcript.
Each flashcard should have:
  - "front": a concise question or key term (<= 1 sentence)
  - "back": a clear answer or definition (<= 2 sentences)

VIDEO TITLE: {video_info.get('title', 'Unknown')}
VIDEO DESCRIPTION: {video_info.get('description', 'No description')}

TRANSCRIPT EXCERPT:
{full_text[:3000]}... [truncated]

Respond **ONLY** in JSON as an array of objects, e.g.:

[
  {{ "front": "What is X?", "back": "X is ..." }},
  ...
]
"""
    raw = await generate_response_with_gemini_async(prompt)

    # 3) Try to parse the JSON array
    try:
        json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw)
        cards = json.loads(json_match.group(0))
    except Exception:
        logger.warning("Failed to parse JSON, using fallback parsing")
        # Fallback: pick out lines prefixed "front:" / "back:"
        cards = []
        cur = {}
        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("front:"):
                if cur:
                    cards.append(cur)
                cur = {"front": line.split(":",1)[1].strip()}
            elif line.lower().startswith("back:") and cur:
                cur["back"] = line.split(":",1)[1].strip()
        if cur.get("back"):
            cards.append(cur)

    return cards

# Example usage:
# flashcards = await generate_flashcards(transcript_segments, video_info)
# Send `flashcards` down to your frontend to render each as a flashcard.
