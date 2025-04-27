# # Add these imports at the top of video_qa.py
# import os
# import re
# import time
# import asyncio
# import json
# from logger_config import logger
# from utils import format_timestamp
# from retrieval import retrieve_chunks
# from highlights import extract_qa_clips, merge_qa_clips
# from constants import OUTPUT_DIR
# import ollama
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import torch
# from functools import lru_cache

# # Initialize sentence transformer model (add this near the top of the file)
# # Use a singleton pattern to load the model only once
# _sentence_transformer = None

# def get_sentence_transformer():
#     """Get or initialize the sentence transformer model."""
#     global _sentence_transformer
#     if _sentence_transformer is None:
#         try:
#             logger.info("Loading sentence transformer model...")
#             # Choose a model - all-MiniLM-L6-v2 is a good balance of speed and accuracy
#             _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
#             logger.info("Sentence transformer model loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading sentence transformer: {e}")
#             logger.info("Falling back to traditional text matching methods")
#             return None
#     return _sentence_transformer

# @lru_cache(maxsize=128)
# def get_text_embedding(text):
#     """Get embedding for text with caching for efficiency."""
#     model = get_sentence_transformer()
#     if model is None:
#         return None
#     try:
#         # Convert text to embedding
#         return model.encode(text)
#     except Exception as e:
#         logger.error(f"Error encoding text: {e}")
#         return None

# def semantic_similarity_match(question, transcript_segments, top_k=10):
#     """Find semantically similar segments using sentence transformers."""
#     model = get_sentence_transformer()
#     if model is None:
#         return []  # Fall back to traditional methods if model isn't available
    
#     try:
#         # Get question embedding
#         question_embedding = get_text_embedding(question)
        
#         # Prepare segment texts and compute embeddings efficiently
#         segment_texts = [seg[2] for seg in transcript_segments]
        
#         # Process in batches to avoid memory issues with large transcripts
#         batch_size = 64
#         all_similarities = []
        
#         for i in range(0, len(segment_texts), batch_size):
#             batch_texts = segment_texts[i:i+batch_size]
#             batch_embeddings = model.encode(batch_texts)
            
#             # Calculate cosine similarity for this batch
#             batch_similarities = np.dot(batch_embeddings, question_embedding) / (
#                 np.linalg.norm(batch_embeddings, axis=1) * np.linalg.norm(question_embedding)
#             )
#             all_similarities.extend(batch_similarities)
        
#         # Create a list of (segment, similarity) pairs
#         segment_similarities = list(zip(transcript_segments, all_similarities))
        
#         # Sort by similarity (highest first) and take top k
#         segment_similarities.sort(key=lambda x: x[1], reverse=True)
        
#         # Return top segments with their scores
#         top_segments = [(seg[0], seg[1], seg[2], float(score)) 
#                         for (seg, score) in segment_similarities[:top_k]
#                         if score > 0.5]  # Only keep reasonably similar segments
        
#         logger.info(f"Found {len(top_segments)} semantically similar segments")
#         return top_segments
        
#     except Exception as e:
#         logger.error(f"Error in semantic similarity matching: {e}")
#         return []  # Fall back to traditional methods if an error occurs

# def find_most_relevant_segments(transcript_segments, question, retrieved_chunks):
#     """Find the most relevant transcript segments for the question using semantic similarity."""
#     import re
#     from difflib import SequenceMatcher
    
#     # Try semantic similarity matching first
#     semantic_matches = semantic_similarity_match(question, transcript_segments)
    
#     if semantic_matches:
#         # Extract segments from semantic matches (ignoring the score)
#         semantic_segments = [(start, end, text) for start, end, text, _ in semantic_matches]
        
#         # Filter out intro segments unless specifically asked about intro
#         if not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
#             semantic_segments = [(start, end, text) for start, end, text in semantic_segments if start >= 30]
        
#         return semantic_segments
    
#     # If semantic matching failed or returned no results, fall back to traditional methods
#     logger.info("Falling back to traditional text matching methods")
    
#     # Extract key terms from the question (excluding stopwords)
#     stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'for', 
#                 'on', 'at', 'by', 'this', 'are', 'or', 'an', 'be', 'as', 'do', 'does', 
#                 'how', 'what', 'when', 'where', 'why', 'who', 'which', 'can', 'could'}
    
#     # Extract quoted phrases first - these are highest priority
#     quoted_phrases = re.findall(r'"([^"]+)"', question)
    
#     # Extract key terms (min 3 chars) and exclude stopwords
#     question_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
#     question_terms = [term for term in question_terms if term not in stopwords]
    
#     # Extract proper nouns (capitalized words) - these are high priority
#     proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
    
#     # Rank segments by relevance with a precision-focused algorithm
#     ranked_segments = []
#     for start, end, text in transcript_segments:
#         text_lower = text.lower()
        
#         # Skip intro segments (first 30 seconds) unless specifically asked about introduction
#         is_intro = start < 30
#         if is_intro and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
#             continue
        
#         # Calculate quoted phrase matches (highest priority)
#         quoted_matches = 0
#         for phrase in quoted_phrases:
#             if phrase.lower() in text_lower:
#                 quoted_matches += 3  # Very high weight
        
#         # Calculate proper noun matches (high priority)
#         proper_matches = 0
#         for noun in proper_nouns:
#             if noun.lower() in text_lower:
#                 proper_matches += 2  # High weight
        
#         # Calculate term frequency match
#         term_matches = sum(1 for term in question_terms if term in text_lower)
        
#         # Calculate similarity ratio
#         similarity = SequenceMatcher(None, question.lower(), text_lower).ratio()
        
#         # Calculate word overlap
#         q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
#         t_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_lower))
#         if len(q_words) > 0:
#             overlap = len(q_words.intersection(t_words)) / len(q_words)  # Focus on question coverage
#         else:
#             overlap = 0
        
#         # Combine scores with a precision-focused weighting
#         total_score = (quoted_matches * 30) + (proper_matches * 20) + (term_matches * 10) + (similarity * 40) + (overlap * 60)
        
#         # Only include segments with a minimum relevance
#         if total_score > 20:  # Higher threshold for more precision
#             ranked_segments.append((start, end, text, total_score))
    
#     # Sort by score (highest first)
#     ranked_segments.sort(key=lambda x: x[3], reverse=True)
    
#     # Take only the most relevant segments
#     top_segments = [seg[:3] for seg in ranked_segments[:5]]  # Limit to top 5
    
#     # If we have at least some good matches, do not fall back to weaker matches
#     if len(top_segments) >= 2:
#         # We have enough good segments
#         pass
#     elif len(top_segments) == 1 and ranked_segments[0][3] > 50:
#         # We have one very strong match
#         pass
#     # Only fall back to retrieved chunks if we don't have strong matches
#     elif not top_segments and retrieved_chunks:
#         logger.info("No direct segment matches, falling back to chunk-based matching")
#         for chunk in retrieved_chunks:
#             chunk_text = chunk.page_content
            
#             # Skip the fallback if it seems to be from the introduction
#             intro_indicators = ['welcome', 'today we', 'in this video', 'going to talk', 'i\'m going to']
#             if any(indicator in chunk_text.lower() for indicator in intro_indicators):
#                 continue
                
#             # Try to find direct matches in transcript segments
#             for start, end, text in transcript_segments:
#                 # Skip intro segments
#                 if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
#                     continue
                    
#                 if text in chunk_text or chunk_text in text:
#                     top_segments.append((start, end, text))
#                     continue
                
#                 # If direct match fails, use sequence matching with higher threshold
#                 similarity = SequenceMatcher(None, chunk_text.lower(), text.lower()).ratio()
#                 if similarity > 0.7:  # 70% similarity threshold for more precision
#                     top_segments.append((start, end, text))
    
#     # Ensure we have sorted, non-duplicate segments
#     unique_segments = []
#     for seg in top_segments:
#         if seg not in unique_segments:
#             unique_segments.append(seg)
    
#     # Sort by timestamp
#     unique_segments.sort(key=lambda x: x[0])
    
#     return unique_segments


# def optimize_clip_segments(segments, transcript_segments, max_gap=5, min_duration=8, max_duration=90):
#     """
#     Optimize clip segments by combining nearby ones and enforcing min/max durations.
#     Ensures clips include complete sentences and natural pauses.
    
#     Args:
#         segments: List of (start, end, text) segments to optimize
#         transcript_segments: Full list of transcript segments for context
#         max_gap: Maximum gap between segments to merge
#         min_duration: Minimum duration for a segment
#         max_duration: Maximum duration for a segment
#     """
#     if not segments:
#         return []
    
#     # Sort segments by start time
#     sorted_segs = sorted(segments, key=lambda x: x[0])
    
#     # Find natural breakpoints in the transcript (sentence ends, pauses)
#     natural_breaks = []
#     for i, (start, end, text) in enumerate(transcript_segments):
#         # Check if this segment ends with a sentence-ending punctuation
#         if re.search(r'[.!?]\s*$', text):
#             natural_breaks.append(end)
        
#         # Check if there's a gap after this segment (indicating a pause)
#         if i < len(transcript_segments) - 1:
#             next_start = transcript_segments[i+1][0]
#             if next_start - end > 0.7:  # Pause of 0.7 seconds or more
#                 natural_breaks.append(end)
    
#     # Combine segments that are close to each other
#     combined = []
#     current_group = [sorted_segs[0]]
    
#     for seg in sorted_segs[1:]:
#         prev_end = current_group[-1][1]
#         current_start = seg[0]
        
#         if current_start - prev_end <= max_gap:
#             # Merge with current group
#             current_group.append(seg)
#         else:
#             # Start a new group
#             start = current_group[0][0]
#             end = current_group[-1][1]
#             text = " ".join([s[2] for s in current_group])
            
#             # Expand to include complete sentences by finding natural breakpoints
#             expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, max_duration)
            
#             # Ensure minimum duration
#             if expanded_end - start < min_duration:
#                 expanded_end = max(start + min_duration, expanded_end)
            
#             # Enforce maximum duration
#             if expanded_end - start > max_duration:
#                 expanded_end = start + max_duration
                
#             combined.append((start, expanded_end, text))
#             current_group = [seg]
    
#     # Add the last group
#     if current_group:
#         start = current_group[0][0]
#         end = current_group[-1][1]
#         text = " ".join([s[2] for s in current_group])
        
#         # Expand to include complete sentences
#         expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, max_duration)
        
#         # Apply duration constraints
#         if expanded_end - start < min_duration:
#             expanded_end = max(start + min_duration, expanded_end)
#         if expanded_end - start > max_duration:
#             expanded_end = start + max_duration
            
#         combined.append((start, expanded_end, text))
    
#     return combined

# def find_natural_endpoint(current_end, natural_breaks, transcript_segments, max_duration, lookahead=15):
#     """
#     Find a natural endpoint for a clip by looking for natural breaks.
    
#     Args:
#         current_end: Current end time of the segment
#         natural_breaks: List of timestamps where natural breaks occur
#         transcript_segments: Full list of transcript segments
#         max_duration: Maximum allowed extension
#         lookahead: How far ahead to look for a natural break (in seconds)
        
#     Returns:
#         Adjusted end time at a natural pause or sentence break
#     """
#     # Find the next natural break after current_end
#     future_breaks = [b for b in natural_breaks if b > current_end and b <= current_end + lookahead]
    
#     if future_breaks:
#         # Find the closest natural break
#         return min(future_breaks)
    
#     # If no natural breaks found, try to find the end of the current sentence in transcript
#     for start, end, text in transcript_segments:
#         # If this segment contains our current endpoint
#         if start <= current_end <= end:
#             # If it ends with sentence-ending punctuation, use the segment end
#             if re.search(r'[.!?]\s*$', text):
#                 return end
            
#             # Otherwise find the next segment that ends a sentence
#             segment_index = transcript_segments.index((start, end, text))
#             for i in range(segment_index+1, min(segment_index+5, len(transcript_segments))):
#                 next_start, next_end, next_text = transcript_segments[i]
#                 if re.search(r'[.!?]\s*$', next_text) and next_end <= current_end + lookahead:
#                     return next_end
    
#     # If no good breakpoint found, just add a small buffer
#     return current_end + 2  # Add 2 seconds buffer


# async def answer_video_question(transcript_segments, video_path, question, full_text=None, generate_clip=True):
#     """
#     Answer a specific question about video content and generate clips with complete thoughts/sentences.
#     Uses semantic matching for finding relevant segments and ensures natural speech boundaries.
    
#     Args:
#         transcript_segments: List of (start_time, end_time, text) tuples
#         video_path: Path to the video file
#         question: User's question about the video
#         full_text: Optional full transcript text
#         generate_clip: Whether to generate a video clip (default: True)
        
#     Returns:
#         Dict containing answer, relevant clip path (if requested), and timestamps
#     """
#     start_time = time.time()
#     logger.info(f"Processing video Q&A for question: {question}")
    
#     # If full text wasn't provided, create it from segments
#     if not full_text:
#         full_text = " ".join([seg[2] for seg in transcript_segments])
    
#     # Step 1: Retrieve the most relevant segments for the question
#     retrieved_chunks = await retrieve_chunks(full_text, question, k=5)
#     retrieved_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    
#     # Step 2: Find the timestamp ranges using semantic similarity matching
#     relevant_segments = find_most_relevant_segments(transcript_segments, question, retrieved_chunks)
    
#     # Skip further processing if we couldn't find any relevant segments
#     if not relevant_segments:
#         logger.warning("No relevant segments found for the question")
#         return {
#             "question": question,
#             "answer": "I'm sorry, but this video doesn't seem to contain information about your question.",
#             "clip_path": None,
#             "formatted_timestamps": [],
#             "clip_title": None,
#             "processing_time": time.time() - start_time
#         }
    
#     # Optimize segments for better clip generation - now includes natural speech boundaries
#     optimized_segments = optimize_clip_segments(relevant_segments, transcript_segments, 
#                                                min_duration=8,   # Longer minimum to include full thoughts
#                                                max_duration=90)  # Longer maximum for complete sentences
    
#     if optimized_segments:
#         relevant_segments = optimized_segments
    
#     # Step 3: Prepare context for the LLM to answer the question
#     timestamps_info = []
#     for start, end, text in relevant_segments:
#         start_fmt = format_timestamp(start)
#         timestamps_info.append(f"[{start_fmt}] {text}")
    
#     context = "\n".join(timestamps_info)
    
#     # Step 4: Generate ONLY the answer and title using Ollama (not timestamps)
#     prompt = f"""
#     Based on the following transcript segments from a video, answer this specific question:
    
#     QUESTION: "{question}"
    
#     VIDEO TRANSCRIPT SEGMENTS:
#     {context}
    
#     Please provide:
#     1. A direct answer to the question based strictly on the video content. If the video doesn't address the question, clearly state that the video doesn't contain the answer.
#     2. A descriptive title for a clip that would answer this question (keep it under 40 characters)
    
#     Format your answer as a JSON object with these fields:
#     {{
#         "answer": "Your detailed answer here...",
#         "clip_title": "Concise descriptive title for this answer clip"
#     }}
    
#     If the video doesn't contain an answer to the question, respond with:
#     {{
#         "answer": "The video doesn't address this question or contain relevant information.",
#         "clip_title": null
#     }}
#     """
    
#     logger.info("Sending question to LLM for analysis...")
#     response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "system", "content": prompt}])
#     raw_content = response["message"]["content"]
    
#     # Extract JSON from response
#     answer_data = None
#     try:
#         # Try to directly parse the response as JSON
#         answer_data = json.loads(raw_content)
#     except json.JSONDecodeError:
#         # Try to extract JSON using regex
#         logger.info("Direct JSON parsing failed, trying regex extraction...")
#         json_match = re.search(r'\{[\s\S]*\}', raw_content)
#         if json_match:
#             try:
#                 answer_data = json.loads(json_match.group(0))
#             except json.JSONDecodeError:
#                 logger.warning("JSON extraction failed")
    
#     # If JSON parsing fails, create a structured response manually
#     if not answer_data:
#         logger.warning("Creating structured answer manually from text response")
#         lines = raw_content.split('\n')
#         answer = ""
#         clip_title = f"Answer to: {question}"
        
#         # Try to extract answer and title from text
#         in_answer = False
#         for line in lines:
#             if not line.strip():
#                 continue
                
#             if line.startswith("1.") or "answer" in line.lower():
#                 in_answer = True
#                 answer = line.replace("1.", "").strip()
#                 continue
#             elif line.startswith("2.") or "title" in line.lower():
#                 in_answer = False
#                 clip_title = line.replace("2.", "").strip()
#             elif in_answer:
#                 answer += " " + line.strip()
        
#         answer_data = {
#             "answer": answer,
#             "clip_title": clip_title
#         }
    
#     # Check if the answer indicates no relevant content
#     if answer_data and "answer" in answer_data:
#         answer_text = answer_data["answer"].lower()
#         no_answer_phrases = [
#             "doesn't address", "does not address", 
#             "doesn't contain", "does not contain", 
#             "no information", "no relevant information",
#             "not mentioned", "doesn't mention", "does not mention",
#             "not discussed", "doesn't discuss", "does not discuss"
#         ]
        
#         if any(phrase in answer_text for phrase in no_answer_phrases):
#             logger.info("LLM determined the video doesn't contain an answer")
#             return {
#                 "question": question,
#                 "answer": answer_data["answer"],
#                 "clip_path": None,
#                 "formatted_timestamps": [],
#                 "clip_title": None,
#                 "processing_time": time.time() - start_time
#             }
    
#     # Step 5: Create highlights directly from the relevant segments (not from LLM)
#     formatted_timestamps = []
#     highlights = []
    
#     for start, end, text in relevant_segments:
#         # Skip segments that are too short or in the intro (unless specifically asked about intro)
#         if end - start < 5:  # Increased minimum to avoid too-short clips
#             continue
            
#         if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
#             continue
            
#         # Format for display
#         start_fmt = format_timestamp(start)
#         end_fmt = format_timestamp(end)
        
#         # Create abbreviated description from the segment text
#         short_desc = text[:80] + "..." if len(text) > 80 else text
#         reason = f"Contains: {short_desc}"
        
#         formatted_timestamps.append(f"{start_fmt} to {end_fmt}: {reason}")
        
#         # Add to highlights for clip generation with expanded context for complete sentences
#         highlights.append({
#             "start": start,
#             "end": end,
#             "description": reason
#         })
    
#     # Ensure we have substantial duration
#     total_highlight_duration = sum(h["end"] - h["start"] for h in highlights)
#     if total_highlight_duration < 5 and highlights:
#         # If highlights are too short, try to expand them
#         for h in highlights:
#             # Find nearby transcript segments to expand this highlight
#             for ts_start, ts_end, ts_text in transcript_segments:
#                 # If this transcript segment is close to our highlight
#                 if abs(ts_start - h["end"]) < 3 or abs(ts_end - h["start"]) < 3:
#                     # Expand highlight to include this segment
#                     h["start"] = min(h["start"], ts_start)
#                     h["end"] = max(h["end"], ts_end)
    
#     # Step 6: Generate video clips (only if requested and we have highlights)
#     clip_path = None
#     if generate_clip and highlights:
#         logger.info(f"Generating {len(highlights)} Q&A clips with complete sentences...")
        
#         # Pass transcript_segments to ensure natural speech boundaries
#         clip_paths, highlight_info = extract_qa_clips(video_path, highlights, transcript_segments)
        
#         if clip_paths:
#             # Merge the clips into a single answer video
#             qa_title = answer_data.get("clip_title", f"Answer: {question}")
#             safe_title = re.sub(r'[^\w\s-]', '', qa_title).strip().replace(' ', '_')
            
#             # Create a directory for Q&A clips
#             qa_dir = os.path.join(OUTPUT_DIR, "qa_clips")
#             os.makedirs(qa_dir, exist_ok=True)
            
#             # Merge all clips into one answer video
#             merged_path = os.path.join(qa_dir, f"{safe_title[:40]}.mp4")
            
#             try:
#                 # If only one clip, just rename it
#                 if len(clip_paths) == 1:
#                     import shutil
#                     shutil.copy(clip_paths[0], merged_path)
#                     clip_path = merged_path
#                 else:
#                     # Merge multiple clips without transitions
#                     clip_path = merge_qa_clips(clip_paths, highlight_info, is_reel=False)
                    
#                     # Copy to the qa_clips directory with the proper name if needed
#                     if clip_path and os.path.exists(clip_path) and clip_path != merged_path:
#                         import shutil
#                         shutil.copy(clip_path, merged_path)
#                         clip_path = merged_path
#             except Exception as e:
#                 logger.error(f"Error merging answer clips: {e}")
                
#                 # Fallback to first clip if merge fails
#                 if clip_paths:
#                     import shutil
#                     try:
#                         shutil.copy(clip_paths[0], merged_path)
#                         clip_path = merged_path
#                     except Exception as copy_e:
#                         logger.error(f"Error copying fallback clip: {copy_e}")
#     else:
#         logger.info("Skipping clip generation as requested or no segments found")
        
#     # Calculate processing time
#     end_time = time.time()
#     processing_time = end_time - start_time
    
#     # Create the final result
#     result = {
#         "question": question,
#         "answer": answer_data.get("answer", "Could not generate an answer from the video content."),
#         "clip_path": clip_path,
#         "formatted_timestamps": formatted_timestamps,
#         "clip_title": answer_data.get("clip_title", f"Answer: {question}"),
#         "processing_time": processing_time
#     }
    
#     # Add diagnostic info for debugging
#     diagnostics = {
#         "num_semantic_segments": len(relevant_segments),
#         "num_highlights": len(highlights),
#         "avg_segment_duration": sum((end-start) for start, end, _ in relevant_segments) / max(1, len(relevant_segments))
#     }
#     result["diagnostics"] = diagnostics
    
#     return result


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
# Import NLTK instead of spaCy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Download necessary NLTK resources if not already present
def ensure_nltk_resources():
    try:
        resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
    except Exception as e:
        logger.error(f"Error ensuring NLTK resources: {e}")

# Run this at startup
ensure_nltk_resources()

# Initialize models
_sentence_transformer = None
_tfidf_vectorizer = None
_nltk_stopwords = None
_lemmatizer = None

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

def get_stopwords():
    """Get NLTK stopwords."""
    global _nltk_stopwords
    if _nltk_stopwords is None:
        try:
            _nltk_stopwords = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            # Fallback stopwords if NLTK fails
            _nltk_stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'for', 
                              'on', 'at', 'by', 'this', 'are', 'or', 'an', 'be', 'as', 'do', 'does', 
                              'how', 'what', 'when', 'where', 'why', 'who', 'which', 'can', 'could'}
    return _nltk_stopwords

def get_lemmatizer():
    """Get WordNet lemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        try:
            _lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.error(f"Error initializing lemmatizer: {e}")
            return None
    return _lemmatizer

def get_tfidf_vectorizer():
    """Get or initialize the TF-IDF vectorizer."""
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        try:
            logger.info("Initializing TF-IDF vectorizer...")
            _tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            logger.info("TF-IDF vectorizer initialized")
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            return None
    return _tfidf_vectorizer

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

def extract_named_entities(text):
    """
    Extract named entities from text using NLTK instead of spaCy.
    Returns a list of (entity_text, entity_type) tuples.
    """
    try:
        # Tokenize and tag the text
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        
        # Use NLTK's named entity chunker
        ne_tree = ne_chunk(tagged_tokens)
        
        # Extract entities
        entities = []
        for subtree in ne_tree:
            if isinstance(subtree, nltk.tree.Tree):
                entity_text = ' '.join([word for word, tag in subtree.leaves()])
                entity_type = subtree.label()
                entities.append((entity_text, entity_type))
        
        return entities
    except Exception as e:
        logger.error(f"Error extracting named entities: {e}")
        return []

def analyze_question_type(question):
    """
    Determine the type of question being asked to better tailor the response.
    Returns a tuple of (question_type, focus_entities)
    """
    question_lower = question.lower()
    
    # Check for question type indicators
    question_types = {
        "what": ["what", "define", "describe", "explain"],
        "how": ["how", "method", "way", "process", "technique"],
        "why": ["why", "reason", "cause", "because"],
        "when": ["when", "time", "date", "period", "during"],
        "where": ["where", "location", "place", "country", "city"],
        "who": ["who", "person", "people", "name"],
        "comparison": ["compare", "difference", "better", "worse", "versus", "vs"],
        "opinion": ["think", "feel", "opinion", "assessment", "perspective", "view"]
    }
    
    # Determine question type
    detected_type = "general"
    for qtype, indicators in question_types.items():
        if any(indicator in question_lower.split() or 
              (f" {indicator} " in question_lower) for indicator in indicators):
            detected_type = qtype
            break
            
    # Extract focus entities using NLTK instead of spaCy
    focus_entities = extract_named_entities(question)
    
    return detected_type, focus_entities

def tokenize_sentences(text):
    """
    Split text into sentences using NLTK.
    """
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing sentences: {e}")
        # Simple fallback sentence splitter
        return re.split(r'(?<=[.!?])\s+', text)

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

def keyword_based_match(question, transcript_segments, top_k=10):
    """Find segments based on TF-IDF keyword matching."""
    try:
        # Get or initialize the TF-IDF vectorizer
        vectorizer = get_tfidf_vectorizer()
        if vectorizer is None:
            return []
            
        # Prepare segment texts
        segment_texts = [seg[2] for seg in transcript_segments]
        all_texts = segment_texts + [question]
        
        # Fit and transform to get TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Get the question vector (last item)
        question_vector = tfidf_matrix[-1]
        
        # Calculate TF-IDF similarity for each segment
        similarities = cosine_similarity(tfidf_matrix[:-1], question_vector)
        
        # Create a list of (segment, similarity) pairs
        segment_similarities = [(seg, float(sim[0])) 
                               for seg, sim in zip(transcript_segments, similarities)]
        
        # Sort by similarity (highest first) and take top k
        segment_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top segments with their scores
        top_segments = [(seg[0], seg[1], seg[2], score) 
                        for (seg, score) in segment_similarities[:top_k]
                        if score > 0.2]  # Only keep reasonably similar segments
        
        logger.info(f"Found {len(top_segments)} keyword-matched segments")
        return top_segments
        
    except Exception as e:
        logger.error(f"Error in keyword-based matching: {e}")
        return []

def entity_based_match(question, transcript_segments, top_k=10):
    """Find segments based on named entity matching, using NLTK instead of spaCy."""
    try:
        # Extract named entities from the question
        question_entities = extract_named_entities(question)
        question_entity_texts = [ent[0].lower() for ent in question_entities]
        
        if not question_entity_texts:
            logger.info("No named entities found in question")
            return []
            
        # Process each segment to find entity matches
        segment_scores = []
        
        for start, end, text in transcript_segments:
            # Skip very short segments
            if len(text.split()) < 3:
                continue
                
            # Extract entities from segment
            seg_entities = extract_named_entities(text)
            seg_entity_texts = [ent[0].lower() for ent in seg_entities]
            
            if not seg_entity_texts:
                continue
                
            # Calculate entity matching score based on exact matches and entity type
            score = 0
            matched_entities = []
            
            for q_ent_text, q_ent_type in question_entities:
                q_ent_text = q_ent_text.lower()
                for s_ent_text, s_ent_type in seg_entities:
                    s_ent_text = s_ent_text.lower()
                    # Exact match with same entity type (highest score)
                    if q_ent_text == s_ent_text and q_ent_type == s_ent_type:
                        score += 1.0
                        matched_entities.append(q_ent_text)
                    # Exact match with different entity type
                    elif q_ent_text == s_ent_text:
                        score += 0.8
                        matched_entities.append(q_ent_text)
                    # Partial match (substring)
                    elif q_ent_text in s_ent_text or s_ent_text in q_ent_text:
                        score += 0.5
                        
            # Normalize score by number of question entities
            if question_entities:
                normalized_score = score / len(question_entities)
            else:
                normalized_score = 0
                
            # Only consider segments with at least one entity match
            if matched_entities:
                segment_scores.append((start, end, text, normalized_score))
        
        # Sort by score (highest first) and take top k
        segment_scores.sort(key=lambda x: x[3], reverse=True)
        top_segments = segment_scores[:top_k]
        
        logger.info(f"Found {len(top_segments)} entity-matched segments")
        return top_segments
        
    except Exception as e:
        logger.error(f"Error in entity-based matching: {e}")
        return []

def detect_topic_from_question(question):
    """Detect the main topic of the question using NLTK instead of spaCy."""
    try:
        # Get stopwords
        stop_words = get_stopwords()
        
        # Extract potential topics: nouns and named entities
        tokens = word_tokenize(question.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns
        nouns = [word for word, tag in tagged if tag.startswith('NN') and word not in stop_words]
        
        # Extract named entities
        entities = extract_named_entities(question)
        entity_texts = [ent[0].lower() for ent in entities]
        
        # Combine and get most frequent
        all_topics = nouns + entity_texts
        if all_topics:
            # Count occurrences
            topic_counts = Counter(all_topics)
            # Return the most common topic
            return topic_counts.most_common(1)[0][0]
        
        # Fallback to simple keyword extraction
        words = [token for token in tokens if token not in stop_words and len(token) > 3]
        
        if words:
            word_counts = Counter(words)
            return word_counts.most_common(1)[0][0]
            
        return None
    
    except Exception as e:
        logger.error(f"Error detecting topic: {e}")
        return None

def hybrid_relevance_matching(question, transcript_segments, retrieved_chunks):
    """
    Combine semantic similarity, keyword-based, and entity-based matching for more accurate results.
    
    Args:
        question: User's question
        transcript_segments: List of (start_time, end_time, text) tuples
        retrieved_chunks: Chunks from the retrieval step
        
    Returns:
        List of (start, end, text) tuples with most relevant segments
    """
    # Get the main topic from question to help with relevance judgments
    main_topic = detect_topic_from_question(question)
    
    # Analyze question type for context-aware scoring
    question_type, focus_entities = analyze_question_type(question)
    logger.info(f"Question type: {question_type}, Main topic: {main_topic}")
    
    # Get matches from each method
    semantic_matches = semantic_similarity_match(question, transcript_segments, top_k=15)
    keyword_matches = keyword_based_match(question, transcript_segments, top_k=15)
    entity_matches = entity_based_match(question, transcript_segments, top_k=10)
    
    # Create a master set of segments
    all_segments = set()
    for matches in [semantic_matches, keyword_matches, entity_matches]:
        for start, end, text, _ in matches:
            all_segments.add((start, end, text))
    
    # Convert to list
    all_segments = list(all_segments)
    
    # Now score all segments using a weighted combination
    segment_scores = []
    
    for start, end, text in all_segments:
        # Initialize score components
        semantic_score = 0.0
        keyword_score = 0.0
        entity_score = 0.0
        exact_match_score = 0.0
        topic_relevance_score = 0.0
        
        # Get semantic score (if available)
        for s_start, s_end, s_text, s_score in semantic_matches:
            if start == s_start and end == s_end:
                semantic_score = s_score
                break
                
        # Get keyword score (if available)
        for k_start, k_end, k_text, k_score in keyword_matches:
            if start == k_start and end == k_end:
                keyword_score = k_score
                break
                
        # Get entity score (if available)
        for e_start, e_end, e_text, e_score in entity_matches:
            if start == e_start and end == e_end:
                entity_score = e_score
                break
        
        # Check for exact phrase matches (quoted phrases)
        quoted_phrases = re.findall(r'"([^"]+)"', question)
        if quoted_phrases:
            text_lower = text.lower()
            for phrase in quoted_phrases:
                if phrase.lower() in text_lower:
                    exact_match_score += 1.0  # High bonus for exact matches
        
        # Topic relevance score - check if the main topic appears in this segment
        if main_topic and main_topic in text.lower():
            topic_relevance_score = 0.5
        
        # Speech context bonuses
        speech_context_score = 0.0
        
        # Check if this is likely a direct answer (question/answer patterns)
        if re.search(r'(?:the answer is|to answer that|answering|in response|to respond)', text.lower()):
            speech_context_score += 0.3
            
        # Time context bonus for "when" questions
        if question_type == "when" and re.search(r'\b(?:in|during|on|at)\s+\d', text):
            speech_context_score += 0.3
            
        # Location context bonus for "where" questions
        if question_type == "where" and re.search(r'\b(?:at|in|on|near|located)\s+(?:the|a)\s+\w+', text):
            speech_context_score += 0.3
        
        # Reason context bonus for "why" questions
        if question_type == "why" and re.search(r'\b(?:because|since|as|due to|reason|why)\b', text.lower()):
            speech_context_score += 0.3
            
        # Explanation context bonus for "how" questions
        if question_type == "how" and re.search(r'\b(?:process|steps|procedure|method|way to)\b', text.lower()):
            speech_context_score += 0.3
        
        # Combine scores with custom weights tailored to question type
        weights = {
            "semantic": 0.35,
            "keyword": 0.25,
            "entity": 0.15,
            "exact_match": 0.15,
            "topic_relevance": 0.05,
            "speech_context": 0.05
        }
        
        # Adjust weights based on question type
        if question_type == "what":
            weights["semantic"] += 0.05
            weights["keyword"] += 0.05
            weights["entity"] -= 0.05
            weights["speech_context"] -= 0.05
        elif question_type == "who":
            weights["entity"] += 0.1
            weights["semantic"] -= 0.05
            weights["keyword"] -= 0.05
        elif question_type == "when" or question_type == "where":
            weights["entity"] += 0.1
            weights["speech_context"] += 0.05
            weights["semantic"] -= 0.1
            weights["keyword"] -= 0.05
        elif question_type == "why" or question_type == "how":
            weights["speech_context"] += 0.1
            weights["semantic"] += 0.05
            weights["entity"] -= 0.1
            weights["keyword"] -= 0.05
            
        # Calculate final weighted score
        final_score = (
            semantic_score * weights["semantic"] +
            keyword_score * weights["keyword"] +
            entity_score * weights["entity"] +
            exact_match_score * weights["exact_match"] +
            topic_relevance_score * weights["topic_relevance"] +
            speech_context_score * weights["speech_context"]
        )
        
        # Position bias - slightly prefer middle sections of video over very beginning
        # This helps avoid generic introductions unless specifically relevant
        position_factor = 1.0
        if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
            position_factor = 0.7  # Reduce score for intro segments
            
        # Apply position factor
        final_score *= position_factor
        
        # Require a minimum score threshold
        if final_score > 0.2:  # Higher threshold for precision
            segment_scores.append((start, end, text, final_score))
    
    # Sort by score and select top segments
    segment_scores.sort(key=lambda x: x[3], reverse=True)
    
    # Limit to reasonable number while ensuring quality
    if len(segment_scores) > 0:
        # If we have a very strong match, be more selective
        if segment_scores[0][3] > 0.8:
            top_segments = segment_scores[:5]  # Take up to 5 with a strong lead match
        else:
            top_segments = segment_scores[:8]  # Take up to 8 with moderate matches
    else:
        top_segments = segment_scores
    
    # Extract just the transcript segments
    result_segments = [(start, end, text) for start, end, text, _ in top_segments]
    
    # If we still don't have matches and have retrieved chunks, use them as fallback
    if not result_segments and retrieved_chunks:
        logger.info("No matches found, falling back to retrieved chunks")
        for chunk in retrieved_chunks:
            chunk_text = chunk.page_content
            
            # Find the most similar transcript segment
            best_match = None
            best_score = 0
            
            for start, end, text in transcript_segments:
                # Skip intro segments unless specifically relevant
                if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
                    continue
                    
                # Calculate similarity
                similarity = SequenceMatcher(None, chunk_text.lower(), text.lower()).ratio()
                
                if similarity > best_score and similarity > 0.5:
                    best_score = similarity
                    best_match = (start, end, text)
            
            if best_match:
                result_segments.append(best_match)
    
    # Sort by timestamp
    result_segments.sort(key=lambda x: x[0])
    
    # Remove duplicates
    unique_segments = []
    for segment in result_segments:
        if segment not in unique_segments:
            unique_segments.append(segment)
    
    return unique_segments

def find_most_relevant_segments(transcript_segments, question, retrieved_chunks):
    """Find the most relevant transcript segments for the question using hybrid matching."""
    # Use hybrid matching approach
    return hybrid_relevance_matching(question, transcript_segments, retrieved_chunks)

def analyze_speech_patterns(transcript_segments):
    """
    Analyze speech patterns to identify natural boundaries, pauses, and emphasis.
    Returns a dict with speech pattern information.
    """
    # Initialize pattern data
    speech_patterns = {
        "natural_breaks": [],      # Timestamps where sentences end or pauses occur
        "emphasis_points": [],     # Timestamps where emphasis likely occurs
        "topic_shifts": [],        # Timestamps where the topic likely changes
        "speech_rate": {}          # Speech rate by segment (words per second)
    }
    
    # Find natural breaks in the transcript (sentence ends, pauses)
    for i, (start, end, text) in enumerate(transcript_segments):
        # Check if this segment ends with a sentence-ending punctuation
        if re.search(r'[.!?]\s*$', text):
            speech_patterns["natural_breaks"].append(end)
        
        # Check if there's a gap after this segment (indicating a pause)
        if i < len(transcript_segments) - 1:
            next_start = transcript_segments[i+1][0]
            if next_start - end > 0.7:  # Pause of 0.7 seconds or more
                speech_patterns["natural_breaks"].append(end)
        
        # Calculate speech rate (words per second)
        duration = end - start
        word_count = len(text.split())
        if duration > 0:
            speech_rate = word_count / duration
            speech_patterns["speech_rate"][(start, end)] = speech_rate
            
            # Very slow or very fast speech might indicate emphasis
            avg_rate = 2.5  # Average speech rate (words per second)
            if speech_rate > avg_rate * 1.5 or speech_rate < avg_rate * 0.6:
                speech_patterns["emphasis_points"].append((start, end))
        
        # Detect topic shifts based on discourse markers
        topic_shift_markers = ["now", "next", "moving on", "turning to", "let's discuss", 
                              "another", "additionally", "furthermore", "moreover", "above all",
                              "first", "second", "third", "finally", "lastly"]
        
        for marker in topic_shift_markers:
            if marker in text.lower():
                speech_patterns["topic_shifts"].append(start)
    
    return speech_patterns

def optimize_clip_segments(segments, transcript_segments, max_gap=5, min_duration=8, max_duration=90):
    """
    Optimize clip segments by combining nearby ones and enforcing min/max durations.
    Ensures clips include complete sentences and natural pauses.
    """
    if not segments:
        return []
    
    # Get speech pattern analysis
    speech_patterns = analyze_speech_patterns(transcript_segments)
    natural_breaks = speech_patterns["natural_breaks"]
    
    # Sort segments by start time
    sorted_segs = sorted(segments, key=lambda x: x[0])
    
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
            expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, 
                                               speech_patterns, max_duration)
            
            # Ensure minimum duration
            if expanded_end - start < min_duration:
                expanded_end = max(start + min_duration, expanded_end)
            
            # Enforce maximum duration while trying to keep complete thoughts
            if expanded_end - start > max_duration:
                # Look for natural breaks within our maximum duration
                breaks_in_range = [b for b in natural_breaks 
                                  if start < b < start + max_duration]
                
                if breaks_in_range:
                    # Find the latest good break point within our max duration
                    expanded_end = max(breaks_in_range)
                else:
                    # If no natural breaks found, use the hard limit
                    expanded_end = start + max_duration
                
            combined.append((start, expanded_end, text))
            current_group = [seg]
    
    # Add the last group
    if current_group:
        start = current_group[0][0]
        end = current_group[-1][1]
        text = " ".join([s[2] for s in current_group])
        
        # Expand to include complete sentences
        expanded_end = find_natural_endpoint(end, natural_breaks, transcript_segments, 
                                           speech_patterns, max_duration)
        
        # Apply duration constraints
        if expanded_end - start < min_duration:
            expanded_end = max(start + min_duration, expanded_end)
        if expanded_end - start > max_duration:
            # Look for natural breaks
            breaks_in_range = [b for b in natural_breaks 
                              if start < b < start + max_duration]
            
            if breaks_in_range:
                expanded_end = max(breaks_in_range)
            else:
                expanded_end = start + max_duration
            
        combined.append((start, expanded_end, text))
    
    # Post-processing: ensure no overlaps and check for very similar segments
    final_segments = []
    for i, (start, end, text) in enumerate(combined):
        # Check if this segment is very similar to any we've already included
        duplicate = False
        for j, (existing_start, existing_end, existing_text) in enumerate(final_segments):
            # If significant overlap and similar content
            time_overlap = min(end, existing_end) - max(start, existing_start)
            time_overlap_percent = time_overlap / min(end - start, existing_end - existing_start)
            
            text_similarity = SequenceMatcher(None, text, existing_text).ratio()
            
            if time_overlap_percent > 0.7 and text_similarity > 0.7:
                duplicate = True
                # Take the longer of the two segments
                if (end - start) > (existing_end - existing_start):
                    final_segments[j] = (start, end, text)
                break
                
        if not duplicate:
            final_segments.append((start, end, text))
    
    # Sort by timestamp
    final_segments.sort(key=lambda x: x[0])
    
    return final_segments

def find_natural_endpoint(current_end, natural_breaks, transcript_segments, speech_patterns, max_duration, lookahead=15):
    """
    Find a natural endpoint for a clip by looking for natural breaks.
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
            # If it ends with sentence-ending punctuation, use the segment end
            if re.search(r'[.!?]\s*$', text):          
                return end
            
            # Otherwise find the next segment that ends a sentence
            try:
                segment_index = transcript_segments.index((start, end, text))
                for i in range(segment_index+1, min(segment_index+5, len(transcript_segments))):
                    next_start, next_end, next_text = transcript_segments[i]
                    if re.search(r'[.!?]\s*$', next_text) and next_end <= current_end + lookahead:
                        return next_end
            except ValueError:
                # Handle case where segment might not be found exactly due to text differences
                pass
    
    # If no good breakpoint found, check for topic shifts or emphasis changes
    topic_shifts = speech_patterns.get("topic_shifts", [])
    future_shifts = [ts for ts in topic_shifts if ts > current_end and ts <= current_end + lookahead]
    
    if future_shifts:
        # End before a new topic starts (good transition point)
        return min(future_shifts)
    
    # If still no good breakpoint found, just add a small buffer
    return current_end + 2  # Add 2 seconds buffer

def extract_noun_chunks(text):
    """
    Extract noun phrases from text using NLTK instead of spaCy's noun_chunks.
    Returns a list of noun phrases.
    """
    try:
        # Tokenize and tag the text
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Simple noun phrase chunking grammar
        grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN.*>+}   # Determiner/possessive, adjectives and nouns
                {<NN.*>+}                   # Sequence of nouns
        """
        
        # Create a chunk parser with our grammar
        chunk_parser = nltk.RegexpParser(grammar)
        
        # Parse the tagged tokens
        tree = chunk_parser.parse(tagged)
        
        # Extract noun phrases
        noun_chunks = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            np_text = ' '.join(word for word, tag in subtree.leaves())
            noun_chunks.append(np_text)
            
        return noun_chunks
        
    except Exception as e:
        logger.error(f"Error extracting noun chunks: {e}")
        return []

def extract_key_claims(transcript_segments, question):
    """
    Extract key claims or statements from transcript segments that directly answer
    the question. This helps identify the most salient parts for the final answer.
    
    Args:
        transcript_segments: List of relevant (start, end, text) segments
        question: The user's question
        
    Returns:
        List of extracted claims with their source segments
    """
    key_claims = []
    
    # Analyze question to determine what we're looking for
    question_type, focus_entities = analyze_question_type(question)
    
    try:
        # Extract key noun phrases from question
        question_nps = extract_noun_chunks(question)
        question_nps = [np.lower() for np in question_nps]
        
        # Extract entities from question
        question_entities = extract_named_entities(question)
        question_entity_texts = [ent[0].lower() for ent in question_entities]
        
        # Process each segment
        for start, end, text in transcript_segments:
            # Skip very short segments
            if len(text.split()) < 5:
                continue
                
            # Split segment into sentences using NLTK
            sentences = tokenize_sentences(text)
            
            for sent in sentences:
                sent_text = sent.strip()
                if len(sent_text.split()) < 3:
                    continue  # Skip very short sentences
                
                # Check if this sentence matches our question type
                is_relevant = False
                
                # For "what" questions, look for definitions or descriptions
                if question_type == "what":
                    # Check for definition patterns
                    if re.search(r'\bis\b|\bare\b|\bmeans\b|\bdefined\b|\brefers to\b', sent_text.lower()):
                        is_relevant = True
                    # Check for noun phrase overlap
                    sent_nps = extract_noun_chunks(sent_text)
                    sent_nps = [np.lower() for np in sent_nps]
                    if any(q_np in " ".join(sent_nps) for q_np in question_nps):
                        is_relevant = True
                
                # For "how" questions, look for process descriptions
                elif question_type == "how":
                    if re.search(r'\bby\b|\bthrough\b|\bsteps?\b|\bprocess\b|\bmethod\b', sent_text.lower()):
                        is_relevant = True
                
                # For "why" questions, look for causal relationships
                elif question_type == "why":
                    if re.search(r'\bbecause\b|\bsince\b|\bdue to\b|\bresult\b|\bcause\b|\breason\b', sent_text.lower()):
                        is_relevant = True
                
                # For "when" questions, look for temporal information
                elif question_type == "when":
                    if re.search(r'\bin\s+\d|\bon\s+\w+\s+\d|\bduring\b|\bwhen\b|\bafter\b|\bbefore\b', sent_text.lower()):
                        is_relevant = True
                
                # For "where" questions, look for location information
                elif question_type == "where":
                    if re.search(r'\bat\s+\w+|\bin\s+\w+|\bnear\b|\blocation\b|\bplace\b|\barea\b', sent_text.lower()):
                        is_relevant = True
                
                # For "who" questions, look for person references
                elif question_type == "who":
                    # Check if the sentence contains person entities
                    sent_entities = extract_named_entities(sent_text)
                    person_entities = [ent[0] for ent in sent_entities if ent[1] == "PERSON"]
                    if person_entities:
                        is_relevant = True
                
                # For comparisons, look for comparative structures
                elif question_type == "comparison":
                    if re.search(r'\bmore\b|\bless\b|\bbetter\b|\bworse\b|\bdifference\b|\bcompare\b|\bthan\b', sent_text.lower()):
                        is_relevant = True
                
                # For general cases or if specific patterns didn't match
                if not is_relevant:
                    # Check entity overlap
                    sent_entities = extract_named_entities(sent_text)
                    sent_entity_texts = [ent[0].lower() for ent in sent_entities]
                    
                    entity_overlap = any(q_ent in " ".join(sent_entity_texts) for q_ent in question_entity_texts)
                    
                    # Check keyword overlap
                    # Extract keywords from question (non-stopwords)
                    stop_words = get_stopwords()
                    lemmatizer = get_lemmatizer()
                    
                    tokens = word_tokenize(question.lower())
                    keywords = []
                    for token in tokens:
                        if token.isalpha() and token not in stop_words and len(token) > 3:
                            if lemmatizer:
                                lemma = lemmatizer.lemmatize(token)
                                keywords.append(lemma)
                            else:
                                keywords.append(token)
                    
                    keyword_overlap = any(keyword in sent_text.lower() for keyword in keywords)
                    
                    is_relevant = entity_overlap or keyword_overlap
                
                # If relevant, add to key claims
                if is_relevant:
                    # Approximate the sentence span within the segment
                    sentence_start = text.find(sent_text) / len(text) * (end - start) if sent_text in text else 0
                    sentence_end = (text.find(sent_text) + len(sent_text)) / len(text) * (end - start) if sent_text in text else 0
                    
                    claim_start = start + sentence_start
                    claim_end = start + sentence_end
                    
                    key_claims.append({
                        "claim": sent_text,
                        "segment": (start, end, text),
                        "claim_span": (claim_start, claim_end)
                    })
        
        # If we found claims, sort by relevance
        if key_claims:
            # Calculate relevance scores
            for claim in key_claims:
                claim_text = claim["claim"]
                
                # Simple relevance score based on question/claim similarity
                score = SequenceMatcher(None, question.lower(), claim_text.lower()).ratio()
                
                # Boost score for direct answers
                if re.search(r'\b(the answer is|is that|to answer|responds)\b', claim_text.lower()):
                    score += 0.2
                
                claim["relevance"] = score
            
            # Sort by relevance
            key_claims.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Return top claims
            return key_claims[:5]
        
        # If no claims were found, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error extracting key claims: {e}")
        return extract_claims_with_patterns(transcript_segments, question)

def extract_claims_with_patterns(transcript_segments, question):
    """Fallback method for claim extraction using simple pattern matching."""
    key_claims = []
    
    # Simple patterns for different question types
    patterns = {
        "what": [r'\bis\b|\bare\b|\bmeans\b|\bdefined\b|\brefers to\b'],
        "how": [r'\bby\b|\bthrough\b|\bsteps?\b|\bprocess\b|\bmethod\b'],
        "why": [r'\bbecause\b|\bsince\b|\bdue to\b|\bresult\b|\bcause\b|\breason\b'],
        "when": [r'\bin\s+\d|\bon\s+\w+\s+\d|\bduring\b|\bwhen\b|\bafter\b|\bbefore\b'],
        "where": [r'\bat\s+\w+|\bin\s+\w+|\bnear\b|\blocation\b|\bplace\b|\barea\b'],
        "who": [r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b']  # Proper names
    }
    
    # Determine question type
    question_type = "general"
    for qtype in patterns.keys():
        if qtype in question.lower():
            question_type = qtype
            break
    
    # Extract key terms from question
    stopwords = get_stopwords()
    
    question_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    question_terms = [term for term in question_terms if term not in stopwords]
    
    # Process each segment
    for start, end, text in transcript_segments:
        # Split into sentences
        sentences = tokenize_sentences(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        for sent in sentences:
            # Check pattern match for question type
            pattern_match = False
            if question_type in patterns:
                for pattern in patterns[question_type]:
                    if re.search(pattern, sent.lower()):
                        pattern_match = True
                        break
            
            # Check term overlap
            term_match = sum(1 for term in question_terms if term in sent.lower())
            
            # Consider relevant if pattern matches or multiple terms match
            if pattern_match or term_match >= 2:
                # Approximate position in segment
                sent_start = text.find(sent) / len(text) * (end - start) if sent in text else 0
                sent_end = (text.find(sent) + len(sent)) / len(text) * (end - start) if sent in text else 0
                
                claim_start = start + sent_start
                claim_end = start + sent_end
                
                key_claims.append({
                    "claim": sent,
                    "segment": (start, end, text),
                    "claim_span": (claim_start, claim_end)
                })
    
    # Calculate simple relevance scores
    for claim in key_claims:
        claim_text = claim["claim"]
        score = SequenceMatcher(None, question.lower(), claim_text.lower()).ratio()
        claim["relevance"] = score
    
    # Sort by relevance
    key_claims.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Return top claims
    return key_claims[:5]

def enhance_answer_quality(question, relevant_segments, extracted_claims=None):
    """
    Enhance the quality of the answer by structuring information from relevant segments
    and extracted claims.
    
    Args:
        question: The user's question
        relevant_segments: List of relevant (start, end, text) segments
        extracted_claims: Optional list of extracted key claims
        
    Returns:
        Enhanced context for the LLM
    """
    # Analyze question to tailor response format
    question_type, focus_entities = analyze_question_type(question)
    
    # Format segments with timestamps
    formatted_segments = []
    for start, end, text in relevant_segments:
        start_fmt = format_timestamp(start)
        formatted_segments.append(f"[{start_fmt}] {text}")
    
    # Base context from segments
    context = "\n".join(formatted_segments)
    
    # If we have extracted claims, enhance with them
    if extracted_claims:
        claim_section = "\nKEY STATEMENTS FROM VIDEO:\n"
        for claim in extracted_claims:
            claim_text = claim["claim"]
            start, end, _ = claim["segment"]
            start_fmt = format_timestamp(start)
            claim_section += f"- [{start_fmt}] {claim_text}\n"
        
        context += claim_section
    
    # Add question-specific guidance
    if question_type == "what":
        context += "\nThis question seeks a definition or description. Focus on clear explanations of concepts."
    elif question_type == "how":
        context += "\nThis question seeks a process or method. Focus on steps, procedures, or techniques."
    elif question_type == "why":
        context += "\nThis question seeks reasons or causes. Focus on explanations of why something occurs or happened."
    elif question_type == "when":
        context += "\nThis question seeks temporal information. Focus on dates, times, or periods mentioned."
    elif question_type == "where":
        context += "\nThis question seeks location information. Focus on places or settings mentioned."
    elif question_type == "who":
        context += "\nThis question seeks information about people. Focus on individuals or groups mentioned."
    elif question_type == "comparison":
        context += "\nThis question seeks a comparison. Focus on similarities, differences, advantages, or disadvantages."
    
    return context

async def answer_video_question(transcript_segments, video_path, question, full_text=None, generate_clip=True):
    """
    Answer a specific question about video content and generate clips with complete thoughts/sentences.
    Uses hybrid matching for finding relevant segments and ensures natural speech boundaries.
    
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
    
    # Step 2: Find the timestamp ranges using hybrid matching approach
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
    
    # Step 3: Extract key claims from relevant segments to better focus the answer
    extracted_claims = extract_key_claims(relevant_segments, question)
    
    # Step 4: Optimize segments for better clip generation - now includes natural speech boundaries
    optimized_segments = optimize_clip_segments(relevant_segments, transcript_segments, 
                                               min_duration=8,   # Longer minimum to include full thoughts
                                               max_duration=90)  # Longer maximum for complete sentences
    
    if optimized_segments:
        relevant_segments = optimized_segments
    
    # Step 5: Prepare enhanced context for the LLM to answer the question
    context = enhance_answer_quality(question, relevant_segments, extracted_claims)
    
    # Step 6: Generate the answer and title using Ollama with improved prompt
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
    
    # Step 7: Create highlights directly from the relevant segments (not from LLM)
    formatted_timestamps = []
    highlights = []
    
    # If we have extracted claims, prioritize segments containing those claims
    if extracted_claims:
        # Convert claim spans to highlight segments
        for claim in extracted_claims:
            claim_start, claim_end = claim["claim_span"]
            start, end, full_text = claim["segment"]
            
            # Ensure reasonable durations
            if end - start < 5:  # Too short
                continue
                
            # Skip intro segments unless specifically relevant
            if start < 30 and not any(term in question.lower() for term in ['intro', 'introduction', 'beginning', 'start']):
                continue
                
            # Format for display
            start_fmt = format_timestamp(start)
            end_fmt = format_timestamp(end)
            
            # Create abbreviated description
            description = f"Contains: {claim['claim'][:80]}..." if len(claim['claim']) > 80 else f"Contains: {claim['claim']}"
            
            formatted_timestamps.append(f"{start_fmt} to {end_fmt}: {description}")
            
            # Add to highlights for clip generation
            highlights.append({
                "start": start,
                "end": end,
                "description": description
            })
    
    # If we don't have enough highlights from claims, add from the relevant segments
    if len(highlights) < 2:
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
            
            # Skip if we already have this timestamp range
            if any(abs(h["start"] - start) < 2 and abs(h["end"] - end) < 2 for h in highlights):
                continue
                
            formatted_timestamps.append(f"{start_fmt} to {end_fmt}: {reason}")
            
            # Add to highlights for clip generation
            highlights.append({
                "start": start,
                "end": end,
                "description": reason
            })
    
    # Ensure we have substantial duration and complete thoughts
    total_highlight_duration = sum(h["end"] - h["start"] for h in highlights)
    if total_highlight_duration < 5 and highlights:
        # If highlights are too short, try to expand them
        speech_patterns = analyze_speech_patterns(transcript_segments)
        
        for h in highlights:
            # Look for natural endpoints to expand to
            expanded_end = find_natural_endpoint(
                h["end"], 
                speech_patterns["natural_breaks"],
                transcript_segments,
                speech_patterns,
                15  # Look ahead up to 15 seconds
            )
            
            if expanded_end > h["end"]:
                h["end"] = expanded_end
            
            # Find nearby transcript segments to expand this highlight
            for ts_start, ts_end, ts_text in transcript_segments:
                # If this transcript segment is close to our highlight
                if abs(ts_start - h["end"]) < 3 or abs(ts_end - h["start"]) < 3:
                    # Expand highlight to include this segment
                    h["start"] = min(h["start"], ts_start)
                    h["end"] = max(h["end"], ts_end)
    
    # Step 8: Generate video clips (only if requested and we have highlights)
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
        "avg_segment_duration": sum((end-start) for start, end, _ in relevant_segments) / max(1, len(relevant_segments)),
        "num_claims_extracted": len(extracted_claims) if extracted_claims else 0,
        "question_type": analyze_question_type(question)[0]
    }
    result["diagnostics"] = diagnostics
    
    return result