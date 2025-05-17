# Add these imports at the top of video_qa.py
import os
import re
import time
import asyncio
import json
from logger_config import logger
from utils import format_timestamp
from retrieval import retrieve_chunks
from highlights import extract_qa_clips, merge_qa_clips
from constants import OUTPUT_DIR
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from functools import lru_cache

# Initialize sentence transformer model (add this near the top of the file)
# Use a singleton pattern to load the model only once
_sentence_transformer = None

def get_sentence_transformer():
    """Get or initialize the sentence transformer model."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            logger.info("Loading sentence transformer model...")
            # Choose a model - all-MiniLM-L6-v2 is a good balance of speed and accuracy
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            logger.info("Falling back to traditional text matching methods")
            return None
    return _sentence_transformer

@lru_cache(maxsize=128)
def get_text_embedding(text):
    """Get embedding for text with caching for efficiency."""
    model = get_sentence_transformer()
    if model is None:
        return None
    try:
        # Convert text to embedding
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return None

def semantic_similarity_match(question, transcript_segments, top_k=10):
    """Find semantically similar segments using sentence transformers."""
    model = get_sentence_transformer()
    if model is None:
        return []  # Fall back to traditional methods if model isn't available
    
    try:
        # Get question embedding
        question_embedding = get_text_embedding(question)
        
        # Prepare segment texts and compute embeddings efficiently
        segment_texts = [seg[2] for seg in transcript_segments]
        
        # Process in batches to avoid memory issues with large transcripts
        batch_size = 64
        all_similarities = []
        
        for i in range(0, len(segment_texts), batch_size):
            batch_texts = segment_texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts)
            
            # Calculate cosine similarity for this batch
            batch_similarities = np.dot(batch_embeddings, question_embedding) / (
                np.linalg.norm(batch_embeddings, axis=1) * np.linalg.norm(question_embedding)
            )
            all_similarities.extend(batch_similarities)
        
        # Create a list of (segment, similarity) pairs
        segment_similarities = list(zip(transcript_segments, all_similarities))
        
        # Sort by similarity (highest first) and take top k
        segment_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top segments with their scores
        top_segments = [(seg[0], seg[1], seg[2], float(score)) 
                        for (seg, score) in segment_similarities[:top_k]
                        if score > 0.5]  # Only keep reasonably similar segments
        
        logger.info(f"Found {len(top_segments)} semantically similar segments")
        return top_segments
        
    except Exception as e:
        logger.error(f"Error in semantic similarity matching: {e}")
        return []  # Fall back to traditional methods if an error occurs

def find_most_relevant_segments(transcript_segments, question, retrieved_chunks):
    """Find the most relevant transcript segments for the question using semantic similarity."""
    import re
    from difflib import SequenceMatcher
    
    # Try semantic similarity matching first
    semantic_matches = semantic_similarity_match(question, transcript_segments)
    
    if semantic_matches:
        # Extract segments from semantic matches (ignoring the score)
        semantic_segments = [(start, end, text) for start, end, text, _ in semantic_matches]
        
        # Filter out intro segments unless specifically asked about intro
        if not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
            semantic_segments = [(start, end, text) for start, end, text in semantic_segments if start >= 30]
        
        return semantic_segments
    
    # If semantic matching failed or returned no results, fall back to traditional methods
    logger.info("Falling back to traditional text matching methods")
    
    # Extract key terms from the question (excluding stopwords)
    stopwords = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 
                 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 
                 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', 
                 "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 
                 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 
                 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', 
                 "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 
                 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 
                 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 
                 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 
                 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 
                 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', 
                 "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', 
                 "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 
                 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 
                 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 
                 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', 
                 "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 
                 'yourself', 'yourselves'}
    
    # Extract quoted phrases first - these are highest priority
    quoted_phrases = re.findall(r'"([^"]+)"', question)
    
    # Extract key terms (min 3 chars) and exclude stopwords
    question_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    question_terms = [term for term in question_terms if term not in stopwords]
    
    # Extract proper nouns (capitalized words) - these are high priority
    proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
    
    # Rank segments by relevance with a precision-focused algorithm
    ranked_segments = []
    for start, end, text in transcript_segments:
        text_lower = text.lower()
        
        # Skip intro segments (first 30 seconds) unless specifically asked about introduction
        is_intro = start < 30
        if is_intro and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
            continue
        
        # Calculate quoted phrase matches (highest priority)
        quoted_matches = 0
        for phrase in quoted_phrases:
            if phrase.lower() in text_lower:
                quoted_matches += 3  # Very high weight
        
        # Calculate proper noun matches (high priority)
        proper_matches = 0
        for noun in proper_nouns:
            if noun.lower() in text_lower:
                proper_matches += 2  # High weight
        
        # Calculate term frequency match
        term_matches = sum(1 for term in question_terms if term in text_lower)
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, question.lower(), text_lower).ratio()
        
        # Calculate word overlap
        q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
        t_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_lower))
        if len(q_words) > 0:
            overlap = len(q_words.intersection(t_words)) / len(q_words)  # Focus on question coverage
        else:
            overlap = 0
        
        # Combine scores with a precision-focused weighting
        total_score = (quoted_matches * 30) + (proper_matches * 20) + (term_matches * 10) + (similarity * 40) + (overlap * 60)
        
        # Only include segments with a minimum relevance
        if total_score > 20:  # Higher threshold for more precision
            ranked_segments.append((start, end, text, total_score))
    
    # Sort by score (highest first)
    ranked_segments.sort(key=lambda x: x[3], reverse=True)
    
    # Take only the most relevant segments
    top_segments = [seg[:3] for seg in ranked_segments[:5]]  # Limit to top 5
    
    # If we have at least some good matches, do not fall back to weaker matches
    if len(top_segments) >= 2:
        # We have enough good segments
        pass
    elif len(top_segments) == 1 and ranked_segments[0][3] > 50:
        # We have one very strong match
        pass
    # Only fall back to retrieved chunks if we don't have strong matches
    elif not top_segments and retrieved_chunks:
        logger.info("No direct segment matches, falling back to chunk-based matching")
        for chunk in retrieved_chunks:
            chunk_text = chunk.page_content
            
            # Skip the fallback if it seems to be from the introduction
            intro_indicators = ['welcome', 'today we', 'in this video', 'going to talk', 'i\'m going to']
            if any(indicator in chunk_text.lower() for indicator in intro_indicators):
                continue
                
            # Try to find direct matches in transcript segments
            for start, end, text in transcript_segments:
                # Skip intro segments
                if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
                    continue
                    
                if text in chunk_text or chunk_text in text:
                    top_segments.append((start, end, text))
                    continue
                
                # If direct match fails, use sequence matching with higher threshold
                similarity = SequenceMatcher(None, chunk_text.lower(), text.lower()).ratio()
                if similarity > 0.7:  # 70% similarity threshold for more precision
                    top_segments.append((start, end, text))
    
    # Ensure we have sorted, non-duplicate segments
    unique_segments = []
    for seg in top_segments:
        if seg not in unique_segments:
            unique_segments.append(seg)
    
    # Sort by timestamp
    unique_segments.sort(key=lambda x: x[0])
    
    return unique_segments


def optimize_clip_segments(segments, transcript_segments, max_gap=5, min_duration=8, max_duration=90):
    """
    Optimize clip segments by combining nearby ones and enforcing min/max durations.
    Ensures clips include complete sentences and natural pauses.
    
    Args:
        segments: List of (start, end, text) segments to optimize
        transcript_segments: Full list of transcript segments for context
        max_gap: Maximum gap between segments to merge
        min_duration: Minimum duration for a segment
        max_duration: Maximum duration for a segment
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segs = sorted(segments, key=lambda x: x[0])
    
    # Find natural breakpoints in the transcript (sentence ends, pauses)
    natural_breaks = []
    for i, (start, end, text) in enumerate(transcript_segments):
        # Check if this segment ends with a sentence-ending punctuation
        if re.search(r'[.!?]\s*$', text):
            natural_breaks.append(end)
        
        # Check if there's a gap after this segment (indicating a pause)
        if i < len(transcript_segments) - 1:
            next_start = transcript_segments[i+1][0]
            if next_start - end > 0.7:  # Pause of 0.7 seconds or more
                natural_breaks.append(end)
    
    # Combine segments that are close to each other
    combined = []
    current_group = [sorted_segs[0]]
    
    for seg in sorted_segs[1:]:
        prev_end = current_group[-1][1]
        current_start = seg[0]
        
        if current_start - prev_end <= max_gap:
            # Merge with current group
            current_group.append(seg)
        else:
            # Start a new group
            start = current_group[0][0]
            end = current_group[-1][1]
            text = " ".join([s[2] for s in current_group])
            
            # Expand to include complete sentences by finding natural breakpoints
            expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, max_duration)
            
            # Ensure minimum duration
            if expanded_end - start < min_duration:
                expanded_end = max(start + min_duration, expanded_end)
            
            # Enforce maximum duration
            if expanded_end - start > max_duration:
                expanded_end = start + max_duration
                
            combined.append((start, expanded_end, text))
            current_group = [seg]
    
    # Add the last group
    if current_group:
        start = current_group[0][0]
        end = current_group[-1][1]
        text = " ".join([s[2] for s in current_group])
        
        # Expand to include complete sentences
        expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, max_duration)
        
        # Apply duration constraints
        if expanded_end - start < min_duration:
            expanded_end = max(start + min_duration, expanded_end)
        if expanded_end - start > max_duration:
            expanded_end = start + max_duration
            
        combined.append((start, expanded_end, text))
    
    return combined

def find_natural_endpoint(current_end, natural_breaks, transcript_segments, max_duration, lookahead=15):
    """
    Find a natural endpoint for a clip by looking for natural breaks.
    
    Args:
        current_end: Current end time of the segment
        natural_breaks: List of timestamps where natural breaks occur
        transcript_segments: Full list of transcript segments
        max_duration: Maximum allowed extension
        lookahead: How far ahead to look for a natural break (in seconds)
        
    Returns:
        Adjusted end time at a natural pause or sentence break
    """
    # Find the next natural break after current_end
    future_breaks = [b for b in natural_breaks if b > current_end and b <= current_end + lookahead]
    
    if future_breaks:
        # Find the closest natural break
        return min(future_breaks)
    
    # If no natural breaks found, try to find the end of the current sentence in transcript
    for start, end, text in transcript_segments:
        # If this segment contains our current endpoint
        if start <= current_end <= end:
            # If it ends with sentence-ending punctuation, use the segment end
            if re.search(r'[.!?]\s*$', text):
                return end
            
            # Otherwise find the next segment that ends a sentence
            segment_index = transcript_segments.index((start, end, text))
            for i in range(segment_index+1, min(segment_index+5, len(transcript_segments))):
                next_start, next_end, next_text = transcript_segments[i]
                if re.search(r'[.!?]\s*$', next_text) and next_end <= current_end + lookahead:
                    return next_end
    
    # If no good breakpoint found, just add a small buffer
    return current_end + 2  # Add 2 seconds buffer


async def answer_video_question(transcript_segments, video_path, question, full_text=None, generate_clip=True):
    """
    Answer a specific question about video content and generate clips with complete thoughts/sentences.
    Uses semantic matching for finding relevant segments and ensures natural speech boundaries.
    
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
    retrieved_chunks = await retrieve_chunks(full_text, question, k=5)
    retrieved_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    
    # Step 2: Find the timestamp ranges using semantic similarity matching
    relevant_segments = find_most_relevant_segments(transcript_segments, question, retrieved_chunks)
    
    # Skip further processing if we couldn't find any relevant segments
    if not relevant_segments:
        logger.warning("No relevant segments found for the question")
        return {
            "question": question,
            "answer": "I'm sorry, but this video doesn't seem to contain information about your question.",
            "clip_path": None,
            "formatted_timestamps": [],
            "clip_title": None,
            "processing_time": time.time() - start_time
        }
    
    # Optimize segments for better clip generation - now includes natural speech boundaries
    optimized_segments = optimize_clip_segments(relevant_segments, transcript_segments, 
                                               min_duration=8,   # Longer minimum to include full thoughts
                                               max_duration=90)  # Longer maximum for complete sentences
    
    if optimized_segments:
        relevant_segments = optimized_segments
    
    # Step 3: Prepare context for the LLM to answer the question
    timestamps_info = []
    for start, end, text in relevant_segments:
        start_fmt = format_timestamp(start)
        timestamps_info.append(f"[{start_fmt}] {text}")
    
    context = "\n".join(timestamps_info)
    
    # Step 4: Generate ONLY the answer and title using Ollama (not timestamps)
    prompt = f"""
    Based on the following transcript segments from a video, answer this specific question:
    
    QUESTION: "{question}"
    
    VIDEO TRANSCRIPT SEGMENTS:
    {context}
    
    Please provide:
    1. A direct answer to the question based strictly on the video content. If the video doesn't address the question, clearly state that the video doesn't contain the answer.
    2. A descriptive title for a clip that would answer this question (keep it under 40 characters)
    
    Format your answer as a JSON object with these fields:
    {{
        "answer": "Your detailed answer here...",
        "clip_title": "Concise descriptive title for this answer clip"
    }}
    
    If the video doesn't contain an answer to the question, respond with:
    {{
        "answer": "The video doesn't address this question or contain relevant information.",
        "clip_title": null
    }}
    """
    
    logger.info("Sending question to LLM for analysis...")
    response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "system", "content": prompt}])
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
        clip_title = f"Answer to: {question}"
        
        # Try to extract answer and title from text
        in_answer = False
        for line in lines:
            if not line.strip():
                continue
                
            if line.startswith("1.") or "answer" in line.lower():
                in_answer = True
                answer = line.replace("1.", "").strip()
                continue
            elif line.startswith("2.") or "title" in line.lower():
                in_answer = False
                clip_title = line.replace("2.", "").strip()
            elif in_answer:
                answer += " " + line.strip()
        
        answer_data = {
            "answer": answer,
            "clip_title": clip_title
        }
    
    # Check if the answer indicates no relevant content
    if answer_data and "answer" in answer_data:
        answer_text = answer_data["answer"].lower()
        no_answer_phrases = [
            "doesn't address", "does not address", 
            "doesn't contain", "does not contain", 
            "no information", "no relevant information",
            "not mentioned", "doesn't mention", "does not mention",
            "not discussed", "doesn't discuss", "does not discuss"
        ]
        
        if any(phrase in answer_text for phrase in no_answer_phrases):
            logger.info("LLM determined the video doesn't contain an answer")
            return {
                "question": question,
                "answer": answer_data["answer"],
                "clip_path": None,
                "formatted_timestamps": [],
                "clip_title": None,
                "processing_time": time.time() - start_time
            }
    
    # Step 5: Create highlights directly from the relevant segments (not from LLM)
    formatted_timestamps = []
    highlights = []
    
    for start, end, text in relevant_segments:
        # Skip segments that are too short or in the intro (unless specifically asked about intro)
        if end - start < 5:  # Increased minimum to avoid too-short clips
            continue
            
        if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
            continue
            
        # Format for display
        start_fmt = format_timestamp(start)
        end_fmt = format_timestamp(end)
        
        # Create abbreviated description from the segment text
        short_desc = text[:80] + "..." if len(text) > 80 else text
        reason = f"Contains: {short_desc}"
        
        formatted_timestamps.append(f"{start_fmt} to {end_fmt}: {reason}")
        
        # Add to highlights for clip generation with expanded context for complete sentences
        highlights.append({
            "start": start,
            "end": end,
            "description": reason
        })
    
    # Ensure we have substantial duration
    total_highlight_duration = sum(h["end"] - h["start"] for h in highlights)
    if total_highlight_duration < 5 and highlights:
        # If highlights are too short, try to expand them
        for h in highlights:
            # Find nearby transcript segments to expand this highlight
            for ts_start, ts_end, ts_text in transcript_segments:
                # If this transcript segment is close to our highlight
                if abs(ts_start - h["end"]) < 3 or abs(ts_end - h["start"]) < 3:
                    # Expand highlight to include this segment
                    h["start"] = min(h["start"], ts_start)
                    h["end"] = max(h["end"], ts_end)
    
    # Step 6: Generate video clips (only if requested and we have highlights)
    clip_path = None
    if generate_clip and highlights:
        logger.info(f"Generating {len(highlights)} Q&A clips with complete sentences...")
        
        # Pass transcript_segments to ensure natural speech boundaries
        clip_paths, highlight_info = extract_qa_clips(video_path, highlights, transcript_segments)
        
        if clip_paths:
            # Merge the clips into a single answer video
            qa_title = answer_data.get("clip_title", f"Answer: {question}")
            safe_title = re.sub(r'[^\w\s-]', '', qa_title).strip().replace(' ', '_')
            
            # Create a directory for Q&A clips
            qa_dir = os.path.join(OUTPUT_DIR, "qa_clips")
            os.makedirs(qa_dir, exist_ok=True)
            
            # Merge all clips into one answer video
            merged_path = os.path.join(qa_dir, f"{safe_title[:40]}.mp4")
            
            try:
                # If only one clip, just rename it
                if len(clip_paths) == 1:
                    import shutil
                    shutil.copy(clip_paths[0], merged_path)
                    clip_path = merged_path
                else:
                    # Merge multiple clips without transitions
                    clip_path = merge_qa_clips(clip_paths, highlight_info, is_reel=False)
                    
                    # Copy to the qa_clips directory with the proper name if needed
                    if clip_path and os.path.exists(clip_path) and clip_path != merged_path:
                        import shutil
                        shutil.copy(clip_path, merged_path)
                        clip_path = merged_path
            except Exception as e:
                logger.error(f"Error merging answer clips: {e}")
                
                # Fallback to first clip if merge fails
                if clip_paths:
                    import shutil
                    try:
                        shutil.copy(clip_paths[0], merged_path)
                        clip_path = merged_path
                    except Exception as copy_e:
                        logger.error(f"Error copying fallback clip: {copy_e}")
    else:
        logger.info("Skipping clip generation as requested or no segments found")
        
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Create the final result
    result = {
        "question": question,
        "answer": answer_data.get("answer", "Could not generate an answer from the video content."),
        "clip_path": clip_path,
        "formatted_timestamps": formatted_timestamps,
        "clip_title": answer_data.get("clip_title", f"Answer: {question}"),
        "processing_time": processing_time
    }
    
    # Add diagnostic info for debugging
    diagnostics = {
        "num_semantic_segments": len(relevant_segments),
        "num_highlights": len(highlights),
        "avg_segment_duration": sum((end-start) for start, end, _ in relevant_segments) / max(1, len(relevant_segments))
    }
    result["diagnostics"] = diagnostics
    
    return result