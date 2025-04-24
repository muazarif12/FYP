import os
import re
import time
import asyncio
import json
from logger_config import logger
from utils import format_timestamp
from retrieval import retrieve_chunks
from highlights import extract_highlights, merge_clips
from constants import OUTPUT_DIR
import ollama

# Add this updated function to video_qa.py

async def answer_video_question(transcript_segments, video_path, question, full_text=None, generate_clip=True):
    """
    Answer a specific question about video content and optionally generate a clip of the relevant part.
    
    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        video_path: Path to the video file
        question: User's question about the video
        full_text: Optional full transcript text
        generate_clip: Whether to generate a video clip (default: True)
        
    Returns:
        Dict containing answer, relevant clip path (if requested), and timestamps
    """
    start_time = time.time()
    logger.info(f"Processing video Q&A for question: {question}")
    
    # If full text wasn't provided, create it from segments
    if not full_text:
        full_text = " ".join([seg[2] for seg in transcript_segments])
    
    # Step 1: Retrieve the most relevant segments for the question
    retrieved_chunks = await retrieve_chunks(full_text, question, k=3)
    retrieved_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    
    # Step 2: Find the timestamp ranges for the retrieved chunks
    relevant_segments = []
    for chunk in retrieved_chunks:
        chunk_text = chunk.page_content
        
        # Find matching segments in the transcript
        matching_segments = []
        for start, end, text in transcript_segments:
            if text in chunk_text or chunk_text in text:
                matching_segments.append((start, end, text))
        
        # If direct match fails, try fuzzy matching with sliding window
        if not matching_segments:
            for i, (start, end, text) in enumerate(transcript_segments):
                # Try a window of 3 segments
                if i + 2 < len(transcript_segments):
                    combined_text = " ".join([transcript_segments[i+j][2] for j in range(3)])
                    if chunk_text in combined_text:
                        for j in range(3):
                            if i+j < len(transcript_segments):
                                s, e, t = transcript_segments[i+j]
                                matching_segments.append((s, e, t))
        
        # Add unique matching segments to our results
        for seg in matching_segments:
            if seg not in relevant_segments:
                relevant_segments.append(seg)
    
    # Sort segments by start time
    relevant_segments.sort(key=lambda x: x[0])
    
    # Step 3: Prepare context for the LLM to answer the question
    if relevant_segments:
        timestamps_info = []
        for start, end, text in relevant_segments:
            start_fmt = format_timestamp(start)
            timestamps_info.append(f"[{start_fmt}] {text}")
        
        context = "\n".join(timestamps_info)
    else:
        context = retrieved_text
    
    # Step 4: Generate the answer using Ollama
    prompt = f"""
    Based on the following transcript segments from a video, answer this question:
    
    QUESTION: "{question}"
    
    VIDEO TRANSCRIPT SEGMENTS:
    {context}
    
    Please provide:
    1. A direct answer to the question based strictly on the video content
    2. The timestamp ranges that are most relevant to this answer
    3. A title for a clip that would answer this question
    
    Format your answer as a JSON object with these fields:
    {{
        "answer": "Your detailed answer here...",
        "relevant_timestamps": [{{
            "start": start_time_in_seconds,
            "end": end_time_in_seconds,
            "reason": "Why this segment answers the question"
        }}],
        "clip_title": "Concise descriptive title for this answer clip"
    }}
    """
    
    logger.info("Sending question to LLM for analysis...")
    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "system", "content": prompt}])
    raw_content = response["message"]["content"]
    
    # Extract JSON from response
    answer_data = None
    try:
        # Try to directly parse the response as JSON
        answer_data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON using regex
        logger.info("Direct JSON parsing failed, trying regex extraction...")
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            try:
                answer_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("JSON extraction failed")
    
    # If JSON parsing fails, create a structured response manually
    if not answer_data:
        logger.warning("Creating structured answer manually from text response")
        lines = raw_content.split('\n')
        answer = ""
        timestamps = []
        clip_title = f"Answer to: {question}"
        
        # Try to extract answer and timestamps from text
        in_answer = False
        for line in lines:
            if line.startswith("1.") or "answer" in line.lower():
                in_answer = True
                answer = line.replace("1.", "").strip()
                continue
            elif line.startswith("2.") or "timestamp" in line.lower():
                in_answer = False
                # Try to extract timestamps with regex
                time_matches = re.findall(r'(\d+:\d+)', line)
                if time_matches and len(time_matches) >= 2:
                    # Convert MM:SS to seconds
                    times = []
                    for tm in time_matches:
                        parts = tm.split(':')
                        if len(parts) == 2:
                            times.append(int(parts[0]) * 60 + int(parts[1]))
                    
                    if len(times) >= 2:
                        timestamps.append({
                            "start": times[0], 
                            "end": times[1],
                            "reason": "Relevant to the question"
                        })
            elif line.startswith("3.") or "title" in line.lower():
                clip_title = line.replace("3.", "").strip()
            elif in_answer:
                answer += " " + line.strip()
        
        # Fallback to relevant segments if no timestamps were found
        if not timestamps and relevant_segments:
            # Use the first and last relevant segment to define a clip range
            start = relevant_segments[0][0]
            end = relevant_segments[-1][1]
            timestamps.append({
                "start": start,
                "end": end,
                "reason": "Contains information relevant to the question"
            })
        
        answer_data = {
            "answer": answer,
            "relevant_timestamps": timestamps,
            "clip_title": clip_title
        }
    
    # Step 5: Generate video clips for the relevant parts (only if requested)
    clip_path = None
    if generate_clip and answer_data and "relevant_timestamps" in answer_data and answer_data["relevant_timestamps"]:
        # Convert timestamp data to highlight format
        highlights = []
        for ts in answer_data["relevant_timestamps"]:
            if "start" in ts and "end" in ts:
                # Ensure the segment is at least 5 seconds long
                start = float(ts["start"])
                end = float(ts["end"])
                if end - start < 5:
                    end = start + 5
                
                highlights.append({
                    "start": start,
                    "end": end,
                    "description": ts.get("reason", "Relevant to question")
                })
        
        # Generate clips if we have valid highlights
        if highlights:
            logger.info(f"Generating {len(highlights)} answer clips...")
            clip_paths, _ = extract_highlights(video_path, highlights)
            
            if clip_paths:
                # Merge the clips into a single answer video
                qa_title = answer_data.get("clip_title", f"Answer: {question}")
                safe_title = re.sub(r'[^\w\s-]', '', qa_title).strip().replace(' ', '_')
                
                # Create a directory for Q&A clips
                qa_dir = os.path.join(OUTPUT_DIR, "qa_clips")
                os.makedirs(qa_dir, exist_ok=True)
                
                # Merge all clips into one answer video
                merged_path = os.path.join(qa_dir, f"{safe_title}.mp4")
                
                try:
                    # If only one clip, just rename it
                    if len(clip_paths) == 1:
                        import shutil
                        shutil.copy(clip_paths[0], merged_path)
                        clip_path = merged_path
                    else:
                        # Merge multiple clips
                        clip_path = merge_clips(clip_paths, highlights, is_reel=False)
                        
                        # Copy to the qa_clips directory with the proper name
                        if clip_path and os.path.exists(clip_path):
                            import shutil
                            shutil.copy(clip_path, merged_path)
                            clip_path = merged_path
                except Exception as e:
                    logger.error(f"Error merging answer clips: {e}")
    else:
        logger.info("Skipping clip generation as requested")
    
    # Format timestamps for display
    formatted_timestamps = []
    if answer_data and "relevant_timestamps" in answer_data:
        for ts in answer_data["relevant_timestamps"]:
            if "start" in ts and "end" in ts:
                start_fmt = format_timestamp(float(ts["start"]))
                end_fmt = format_timestamp(float(ts["end"]))
                reason = ts.get("reason", "Relevant segment")
                formatted_timestamps.append(f"{start_fmt} to {end_fmt}: {reason}")
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "question": question,
        "answer": answer_data.get("answer", "Could not generate an answer from the video content."),
        "clip_path": clip_path,
        "formatted_timestamps": formatted_timestamps,
        "clip_title": answer_data.get("clip_title", f"Answer: {question}"),
        "processing_time": processing_time
    }

async def generate_faq(transcript_segments, video_info):
    """
    Generate common questions and answers about the video content.
    
    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        video_info: Dictionary with video metadata
        
    Returns:
        List of question-answer pairs
    """
    # Extract full text from transcript
    full_text = " ".join([seg[2] for seg in transcript_segments])
    
    # Create a prompt for generating FAQ
    prompt = f"""
    Based on this video transcript, generate 5 frequently asked questions (FAQ) that viewers might have.
    
    VIDEO TITLE: {video_info.get('title', 'Unknown')}
    VIDEO DESCRIPTION: {video_info.get('description', 'No description')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:3000]}... [truncated]
    
    For each question:
    1. The question should be specific to the content
    2. It should target important or interesting information from the video
    3. Provide a concise, accurate answer based solely on the transcript
    
    Return the results in this JSON format:
    {{
        "faq": [
            {{
                "question": "First question here?",
                "answer": "Answer to first question"
            }},
            ...
        ]
    }}
    """
    
    logger.info("Generating FAQ for video content...")
    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "system", "content": prompt}])
    raw_content = response["message"]["content"]
    
    # Extract JSON from response
    faq_data = None
    try:
        # Try to directly parse the response as JSON
        faq_data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON using regex
        logger.info("Direct JSON parsing failed, trying regex extraction...")
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            try:
                faq_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("JSON extraction failed")
    
    # If JSON parsing fails, create a structured response manually
    if not faq_data or "faq" not in faq_data:
        logger.warning("Creating structured FAQ manually from text response")
        faq_list = []
        
        # Try to extract Q&A pairs from text
        lines = raw_content.split('\n')
        current_question = None
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with number followed by question mark
            q_match = re.match(r'^(\d+[\)\.])?\s*(.+\?)', line)
            if q_match:
                # Save previous Q&A pair if exists
                if current_question and current_answer:
                    faq_list.append({
                        "question": current_question,
                        "answer": current_answer.strip()
                    })
                    current_answer = ""
                
                # New question
                current_question = q_match.group(2).strip()
            elif current_question and not current_answer and ":" in line:
                # Handle "Q: question" format
                parts = line.split(":", 1)
                if parts[0].strip().lower() in ["q", "question"]:
                    current_question = parts[1].strip()
                elif parts[0].strip().lower() in ["a", "answer"]:
                    current_answer = parts[1].strip()
            elif current_question:
                # Collect answer lines
                current_answer += " " + line
        
        # Add the last Q&A pair
        if current_question and current_answer:
            faq_list.append({
                "question": current_question,
                "answer": current_answer.strip()
            })
        
        faq_data = {"faq": faq_list}
    
    return faq_data.get("faq", [])