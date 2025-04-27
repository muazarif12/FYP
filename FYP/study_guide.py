from asyncio.log import logger
import json
import re

import ollama


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