# # import os
# # import time
# # from logger_config import logger
# # from utils import format_time_duration, format_timestamp
# # from highlights import extract_highlights, merge_clips, snap_highlights_to_transcript_boundaries, validate_highlight_quality

# # def convert_timestamp_to_seconds(timestamp):
# #     """Convert HH:MM:SS timestamp to seconds."""
# #     parts = timestamp.split(":")
# #     if len(parts) == 3:
# #         hours, minutes, seconds = map(int, parts)
# #         return hours * 3600 + minutes * 60 + seconds
# #     elif len(parts) == 2:
# #         minutes, seconds = map(int, parts)
# #         return minutes * 60 + seconds
# #     return 0

# # async def generate_highlights_algorithmically(video_path, transcript_segments, video_info, 
# #                                              target_duration=None, is_reel=False):
# #     """
# #     Generate highlights based on algorithmically detected key moments without any LLM usage.
    
# #     Args:
# #         video_path: Path to the video file
# #         transcript_segments: List of transcript segments with timestamps
# #         video_info: Dictionary with video title and description
# #         target_duration: Target duration for highlights in seconds
# #         is_reel: Whether to optimize for short-form content
        
# #     Returns:
# #         Tuple of (output_path, highlight_segments)
# #     """
# #     start_time = time.time()
# #     print(f"Generating {'reel' if is_reel else 'highlights'} algorithmically...")
    
# #     # Get video duration
# #     from moviepy import VideoFileClip
# #     with VideoFileClip(video_path) as video:
# #         video_duration = video.duration
    
# #     # Calculate target highlight parameters
# #     if is_reel:
# #         max_highlight_duration = 60  # 1 minute max for reels
# #         min_segment_duration = 5     # 5 seconds minimum per segment
# #         max_segment_duration = 15    # 15 seconds maximum per segment
# #     elif target_duration:
# #         max_highlight_duration = target_duration
# #         min_segment_duration = 8
# #         max_segment_duration = min(30, target_duration / 3)
# #     else:
# #         # Scale based on video length
# #         if video_duration < 300:  # < 5 min videos
# #             max_highlight_duration = video_duration * 0.4  # 40% of original
# #         elif video_duration < 900:  # 5-15 min videos
# #             max_highlight_duration = video_duration * 0.25  # 25% of original
# #         else:  # > 15 min videos
# #             max_highlight_duration = video_duration * 0.15  # 15% of original
            
# #         max_highlight_duration = min(300, max_highlight_duration)  # Cap at 5 minutes
# #         min_segment_duration = 10
# #         max_segment_duration = 30
    
# #     # Create highlight segments directly without LLM
# #     highlights = []
    
# #     # Always include intro at start
# #     intro_duration = min(30, max_segment_duration)
# #     highlights.append({
# #         "start": 0,
# #         "end": intro_duration,
# #         "description": "Introduction"
# #     })
    
# #     # Create evenly spaced segments throughout the video
# #     # Calculate how many segments needed based on target duration
# #     if target_duration:
# #         num_segments = max(3, int(target_duration / max_segment_duration))
# #     else:
# #         # Number of segments based on video duration
# #         if video_duration < 300:  # < 5 min
# #             num_segments = 3
# #         elif video_duration < 600:  # 5-10 min
# #             num_segments = 5
# #         elif video_duration < 1200:  # 10-20 min
# #             num_segments = 7
# #         else:  # > 20 min
# #             num_segments = max(7, min(10, int(video_duration / 180)))  # One segment per 3 minutes
    
# #     # Always add one for conclusion segment
# #     num_middle_segments = num_segments - 2  # Account for intro and conclusion
    
# #     # Generate evenly spaced segments (middle segments)
# #     if num_middle_segments > 0 and video_duration > 60:
# #         usable_duration = video_duration - (2 * max_segment_duration)  # Remove intro and outro time
# #         for i in range(num_middle_segments):
# #             # Calculate position with slight randomization for natural feeling
# #             position_ratio = (i + 1) / (num_middle_segments + 1)
# #             segment_start = intro_duration + (usable_duration * position_ratio)
            
# #             # Add slight jitter for more natural spacing
# #             import random
# #             jitter = random.uniform(-0.05, 0.05) * max_segment_duration
# #             segment_start = max(intro_duration, min(video_duration - max_segment_duration, segment_start + jitter))
            
# #             # Create segment
# #             highlights.append({
# #                 "start": segment_start,
# #                 "end": segment_start + max_segment_duration,
# #                 "description": f"Key moment {i+1}"
# #             })
    
# #     # Always include ending/conclusion
# #     if video_duration > 60:
# #         outro_start = max(0, video_duration - min(30, max_segment_duration))
# #         highlights.append({
# #             "start": outro_start,
# #             "end": video_duration,
# #             "description": "Conclusion"
# #         })
    
# #     # Sort by start time and ensure no overlaps
# #     highlights.sort(key=lambda x: x["start"])
# #     for i in range(1, len(highlights)):
# #         if highlights[i]["start"] < highlights[i-1]["end"]:
# #             highlights[i]["start"] = highlights[i-1]["end"]
# #             # Ensure segment isn't too short after adjustment
# #             if highlights[i]["end"] - highlights[i]["start"] < min_segment_duration:
# #                 highlights[i]["end"] = highlights[i]["start"] + min_segment_duration
    
# #     # If we have transcript segments, try to snap to speaker boundaries
# #     if transcript_segments:
# #         # Convert transcript_segments to format needed by snap function
# #         transcript_for_snap = []
# #         for start, end, text in transcript_segments:
# #             transcript_for_snap.append({"start": start, "end": end, "text": text})
        
# #         # Snap to transcript boundaries for more natural cuts
# #         highlights = snap_highlights_to_transcript_boundaries(highlights, transcript_for_snap, max_shift=2.0)
    
# #     # Check if we're within target duration
# #     total_duration = sum(h["end"] - h["start"] for h in highlights)
    
# #     # If we're over target duration, trim segments proportionally
# #     if target_duration and total_duration > target_duration * 1.1:  # Allow 10% over
# #         excess_ratio = target_duration / total_duration
        
# #         # Never trim intro or conclusion
# #         intro_segment = highlights[0]
# #         conclusion_segment = highlights[-1]
# #         middle_segments = highlights[1:-1]
        
# #         # Calculate how much we need to reduce middle segments
# #         middle_duration = sum(h["end"] - h["start"] for h in middle_segments)
# #         target_middle_duration = target_duration - (intro_segment["end"] - intro_segment["start"]) - (conclusion_segment["end"] - conclusion_segment["start"])
        
# #         if middle_duration > 0 and target_middle_duration > 0:
# #             reduction_ratio = target_middle_duration / middle_duration
            
# #             # Adjust each middle segment
# #             for h in middle_segments:
# #                 segment_duration = h["end"] - h["start"]
# #                 new_duration = max(min_segment_duration, segment_duration * reduction_ratio)
# #                 h["end"] = h["start"] + new_duration
    
# #     # Extract and merge clips
# #     logger.info("Extracting highlight clips...")
# #     clip_paths, highlight_info = extract_highlights(video_path, highlights)
    
# #     logger.info("Merging clips into final video...")
# #     output_path = merge_clips(clip_paths, highlight_info, is_reel=is_reel)
    
# #     end_time = time.time()
# #     logger.info(f"Algorithmic highlights generation completed in {end_time - start_time:.2f} seconds")
    
# #     return output_path, highlights


# import os
# import time
# import re
# import math
# import numpy as np
# from collections import Counter
# from itertools import groupby
# from logger_config import logger
# from utils import format_time_duration, format_timestamp
# from highlights import extract_highlights, merge_clips, snap_highlights_to_transcript_boundaries

# def convert_timestamp_to_seconds(timestamp):
#     """Convert HH:MM:SS timestamp to seconds."""
#     parts = timestamp.split(":")
#     if len(parts) == 3:
#         hours, minutes, seconds = map(int, parts)
#         return hours * 3600 + minutes * 60 + seconds
#     elif len(parts) == 2:
#         minutes, seconds = map(int, parts)
#         return minutes * 60 + seconds
#     return 0

# def get_video_type_from_metadata(video_info):
#     """Determine the likely video type based on title and description."""
#     title = video_info.get('title', '').lower()
#     desc = video_info.get('description', '').lower()
    
#     # Check for various video types
#     if any(term in title or term in desc for term in 
#            ['lecture', 'lesson', 'tutorial', 'education', 'learn', 'course', 'teach', 'study']):
#         return "educational"
    
#     elif any(term in title or term in desc for term in 
#             ['game', 'gameplay', 'play', 'match', 'tournament', 'sport', 'versus', 'vs']):
#         return "gaming_sports"
    
#     elif any(term in title or term in desc for term in 
#             ['vlog', 'day in', 'travel', 'trip', 'journey', 'experience', 'daily']):
#         return "vlog"
    
#     elif any(term in title or term in desc for term in 
#             ['review', 'unbox', 'product', 'testing', 'compared', 'comparison']):
#         return "review"
    
#     elif any(term in title or term in desc for term in 
#             ['news', 'report', 'update', 'coverage', 'headlines', 'current events']):
#         return "news"
            
#     elif any(term in title or term in desc for term in 
#             ['interview', 'podcast', 'discussion', 'talk', 'conversation']):
#         return "interview"
            
#     elif any(term in title or term in desc for term in 
#             ['comedy', 'funny', 'humor', 'sketch', 'joke', 'prank']):
#         return "entertainment"
            
#     elif any(term in title or term in desc for term in 
#             ['how to', 'diy', 'guide', 'tips', 'tricks', 'tutorial', 'hack']):
#         return "how_to"
    
#     # Default to general
#     return "general"

# def extract_key_phrases(text, max_phrases=3):
#     """Extract key phrases from text using tf-idf like approach."""
#     # Clean and tokenize text
#     text = re.sub(r'[^\w\s]', '', text.lower())
#     words = text.split()
    
#     # Stopwords to ignore
#     stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 
#                 'as', 'for', 'on', 'was', 'by', 'be', 'at', 'this', 'are', 'or',
#                 'an', 'but', 'not', 'you', 'from', 'have', 'we', 'they', 'i', 'so',
#                 'there', 'what', 'can', 'all', 'which', 'when', 'been', 'would', 
#                 'just', 'about', 'some', 'will', 'very', 'also', 'like'}
    
#     # Remove stopwords
#     filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
#     # Count word frequencies
#     word_counts = Counter(filtered_words)
    
#     # Extract phrases (run of 2-3 consecutive meaningful words)
#     phrases = []
#     for i in range(len(filtered_words) - 1):
#         if i < len(filtered_words) - 2:
#             trigram = ' '.join(filtered_words[i:i+3])
#             phrases.append(trigram)
        
#         bigram = ' '.join(filtered_words[i:i+2])
#         phrases.append(bigram)
    
#     # Score phrases based on word importance
#     phrase_scores = {}
#     for phrase in phrases:
#         if phrase not in phrase_scores:
#             words_in_phrase = phrase.split()
#             # Score is sum of word frequencies
#             score = sum(word_counts[w] for w in words_in_phrase)
#             # Multiply by length for slight bonus to longer phrases
#             score *= math.sqrt(len(words_in_phrase))
#             phrase_scores[phrase] = score
    
#     # Get top phrases
#     top_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
#     return [phrase for phrase, score in top_phrases[:max_phrases]]

# def find_topic_transitions(transcript_segments, min_distance=30):
#     """Identify potential topic transitions in the transcript."""
#     transitions = []
#     segment_texts = []
#     segment_starts = []
    
#     # Extract all text segments and their start times
#     for start, _, text in transcript_segments:
#         segment_texts.append(text)
#         segment_starts.append(start)
    
#     # Need at least several segments to find transitions
#     if len(segment_texts) < 10:
#         return transitions
    
#     # Create simplified bag-of-words for each segment
#     bow_segments = []
#     for text in segment_texts:
#         # Clean and tokenize
#         text = re.sub(r'[^\w\s]', '', text.lower())
#         words = text.split()
        
#         # Remove stopwords
#         stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it'}
#         words = [w for w in words if w not in stopwords and len(w) > 2]
        
#         # Create bag of words
#         bow_segments.append(Counter(words))
    
#     # Detect topic shifts by comparing adjacent windows
#     window_size = 3  # Compare blocks of segments
#     for i in range(window_size, len(bow_segments) - window_size):
#         # Previous window
#         prev_window = Counter()
#         for j in range(i - window_size, i):
#             prev_window.update(bow_segments[j])
            
#         # Next window
#         next_window = Counter()
#         for j in range(i, i + window_size):
#             next_window.update(bow_segments[j])
            
#         # Calculate similarity
#         common_words = set(prev_window.keys()) & set(next_window.keys())
#         if not common_words:
#             similarity = 0
#         else:
#             # Cosine similarity-inspired measure
#             prev_sum = sum(prev_window[w]**2 for w in prev_window)
#             next_sum = sum(next_window[w]**2 for w in next_window)
#             common_sum = sum(prev_window[w] * next_window[w] for w in common_words)
            
#             similarity = common_sum / (math.sqrt(prev_sum) * math.sqrt(next_sum))
        
#         # Low similarity indicates topic change
#         if similarity < 0.3:  # Threshold for topic change
#             transitions.append((segment_starts[i], segment_texts[i], similarity))
    
#     # Filter transitions that are too close to each other
#     filtered_transitions = []
#     for i, (start, text, sim) in enumerate(transitions):
#         # Check if this transition is far enough from previous ones
#         if i == 0 or start - transitions[i-1][0] >= min_distance:
#             filtered_transitions.append((start, text, sim))
    
#     return filtered_transitions

# def detect_energy_changes(transcript_segments, video_duration):
#     """Detect segments with high energy based on speech patterns."""
#     energy_points = []
    
#     for i, (start, end, text) in enumerate(transcript_segments):
#         segment_duration = end - start
#         word_count = len(text.split())
        
#         # Skip very short or empty segments
#         if segment_duration < 1 or word_count < 2:
#             continue
        
#         energy = 0
        
#         # Speech rate
#         words_per_second = word_count / segment_duration
#         if words_per_second > 3:  # Fast speech
#             energy += 5
        
#         # Punctuation as energy indicator
#         exclamations = text.count('!')
#         questions = text.count('?')
#         energy += exclamations * 3
#         energy += questions * 2
        
#         # Special phrases indicating emphasis
#         emphasis_phrases = ["important", "key point", "remember", "note that", 
#                            "critical", "crucial", "significant", "essential"]
#         for phrase in emphasis_phrases:
#             if phrase in text.lower():
#                 energy += 10
                
#         # All caps words indicate emphasis
#         caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
#         energy += len(caps_words) * 3
        
#         # Check for numeric facts (numbers often indicate important data points)
#         numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
#         energy += len(numbers) * 2
        
#         # If this segment has significant energy, add it to our points
#         if energy > 5:
#             energy_points.append({
#                 "start": start,
#                 "end": end,
#                 "text": text,
#                 "energy": energy
#             })
    
#     # Sort by energy level
#     energy_points.sort(key=lambda x: x["energy"], reverse=True)
#     return energy_points

# def analyze_transcript_advanced(transcript_segments, video_info, video_duration):
#     """Advanced transcript analysis combining multiple techniques."""
#     segments = []
    
#     # Convert transcript segments to more usable format
#     for start, end, text in transcript_segments:
#         segments.append({
#             "start": start,
#             "end": end,
#             "text": text,
#             "importance": 0  # Will be calculated
#         })
    
#     # Get video type
#     video_type = get_video_type_from_metadata(video_info)
    
#     # Find topic transitions
#     transitions = find_topic_transitions(transcript_segments)
#     transition_times = [t[0] for t in transitions]
    
#     # Find energy points
#     energy_points = detect_energy_changes(transcript_segments, video_duration)
#     energy_times = [p["start"] for p in energy_points]
    
#     # Calculate overall importance score for each segment
#     for i, segment in enumerate(segments):
#         importance = 0
#         text = segment["text"].lower()
#         start_time = segment["start"]
#         end_time = segment["end"]
        
#         # Position-based importance (intro/conclusion)
#         if start_time <= video_duration * 0.05:  # First 5%
#             importance += 30
#         elif start_time >= video_duration * 0.9:  # Last 10%
#             importance += 25
        
#         # Video-type specific weights
#         if video_type == "educational":
#             # In educational videos, explanations, definitions, examples are important
#             if any(term in text for term in ["example", "definition", "means", "is called", 
#                                            "defined as", "this is", "for instance"]):
#                 importance += 20
                
#             # Numerical information is often important in educational content
#             numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
#             importance += len(numbers) * 5
            
#         elif video_type == "gaming_sports":
#             # In gaming/sports, exciting moments and results are important
#             if any(term in text for term in ["win", "score", "point", "goal", "shot", 
#                                            "amazing", "incredible", "victory"]):
#                 importance += 25
                
#         elif video_type == "vlog":
#             # In vlogs, transitions between locations and activities are important
#             if any(term in text for term in ["now we're", "here we are", "next", "then",
#                                            "after that", "arrived", "going to"]):
#                 importance += 15
                
#         elif video_type == "review":
#             # In reviews, ratings, pros/cons, and conclusions are important
#             if any(term in text for term in ["rating", "recommend", "score", "stars",
#                                            "pros", "cons", "verdict", "conclusion"]):
#                 importance += 20
                
#         elif video_type == "how_to":
#             # In how-to videos, steps and warnings are important
#             if any(term in text for term in ["step", "next", "first", "then", "finally",
#                                            "warning", "caution", "important to note"]):
#                 importance += 20
        
#         # Topic transition bonus (segment starts near a detected topic change)
#         for trans_time in transition_times:
#             if abs(start_time - trans_time) < 5:  # Within 5 seconds
#                 importance += 30
#                 break
        
#         # Energy level bonus
#         for energy_time in energy_times:
#             if abs(start_time - energy_time) < 3:  # Within 3 seconds
#                 # Find the energy point and add its energy score
#                 for point in energy_points:
#                     if point["start"] == energy_time:
#                         importance += point["energy"]
#                         break
#                 break
        
#         # Extract key phrases for semantic understanding
#         key_phrases = extract_key_phrases(text)
#         importance += len(key_phrases) * 5
        
#         # Content-based importance
#         emphasis_terms = ["important", "key", "critical", "essential", "vital", 
#                          "remember", "note that", "crucial", "significant", "main"]
#         conclusion_terms = ["in conclusion", "to summarize", "in summary", "finally", 
#                            "to conclude", "wrapping up", "to finish", "lastly"]
#         highlight_terms = ["highlight", "takeaway", "ultimately", "point", "focus"]
        
#         for term in emphasis_terms:
#             if term in text:
#                 importance += 20
        
#         for term in conclusion_terms:
#             if term in text:
#                 importance += 15
        
#         for term in highlight_terms:
#             if term in text:
#                 importance += 10
                
#         # Duration bonus - slightly favor longer segments that have complete thoughts
#         segment_duration = end_time - start_time
#         ideal_duration = 20  # Around 20 seconds is often good for a highlight segment
#         duration_factor = 1 - abs(segment_duration - ideal_duration) / ideal_duration
#         duration_factor = max(0, duration_factor)  # Keep it non-negative
#         importance += duration_factor * 10
        
#         # Store calculated importance
#         segments[i]["importance"] = importance
    
#     # Sort segments by importance
#     segments.sort(key=lambda x: x["importance"], reverse=True)
    
#     return segments

# def select_highlights_with_coverage(important_segments, video_duration, target_num_segments, 
#                                   intro_duration, outro_start):
#     """Select highlights balancing importance with good video coverage."""
#     highlights = []
    
#     # Create time buckets for even distribution
#     num_buckets = max(3, target_num_segments-2)  # Excluding intro and conclusion
#     bucket_duration = (outro_start - intro_duration) / num_buckets
#     buckets = [[] for _ in range(num_buckets)]
    
#     # Assign segments to buckets
#     for segment in important_segments:
#         if segment["start"] < intro_duration or segment["start"] >= outro_start:
#             continue  # Skip segments in intro or conclusion range
            
#         bucket_index = min(num_buckets-1, int((segment["start"] - intro_duration) / bucket_duration))
#         buckets[bucket_index].append(segment)
    
#     # Sort segments within each bucket by importance
#     for bucket in buckets:
#         bucket.sort(key=lambda x: x["importance"], reverse=True)
    
#     # Take the most important segment from each bucket
#     segments_per_bucket = 1
#     segments_to_select = min(target_num_segments-2, sum(len(b) > 0 for b in buckets))
    
#     # If we have more buckets than segments to select, prioritize the buckets with 
#     # the most important segments
#     if segments_to_select < len(buckets):
#         # Get the most important segment from each bucket
#         best_from_each = []
#         for i, bucket in enumerate(buckets):
#             if bucket:
#                 best_from_each.append((i, bucket[0]["importance"]))
        
#         # Sort buckets by the importance of their best segment
#         best_from_each.sort(key=lambda x: x[1], reverse=True)
#         selected_bucket_indices = [idx for idx, _ in best_from_each[:segments_to_select]]
#     else:
#         selected_bucket_indices = range(len(buckets))
    
#     # Select segments from chosen buckets
#     for bucket_idx in selected_bucket_indices:
#         bucket = buckets[bucket_idx]
#         if not bucket:
#             continue
            
#         # Take up to segments_per_bucket from this bucket
#         for i in range(min(segments_per_bucket, len(bucket))):
#             segment = bucket[i]
#             highlights.append({
#                 "start": segment["start"],
#                 "end": segment["end"],
#                 "description": f"Key moment: {segment['text'][:50]}..." if len(segment['text']) > 50 else f"Key moment: {segment['text']}"
#             })
    
#     return highlights

# async def generate_highlights_algorithmically(video_path, transcript_segments, video_info, 
#                                              target_duration=None, is_reel=False):
#     """
#     Generate highlights using advanced algorithmic analysis without LLM.
    
#     Args:
#         video_path: Path to the video file
#         transcript_segments: List of transcript segments with timestamps
#         video_info: Dictionary with video title and description
#         target_duration: Target duration for highlights in seconds
#         is_reel: Whether to optimize for short-form content
        
#     Returns:
#         Tuple of (output_path, highlight_segments)
#     """
#     start_time = time.time()
#     print(f"Generating {'reel' if is_reel else 'highlights'} using advanced analysis...")
    
#     # Get video duration
#     from moviepy import VideoFileClip
#     with VideoFileClip(video_path) as video:
#         video_duration = video.duration
    
#     # Calculate target highlight parameters
#     if is_reel:
#         max_highlight_duration = 60  # 1 minute max for reels
#         min_segment_duration = 5     # 5 seconds minimum per segment
#         max_segment_duration = 15    # 15 seconds maximum per segment
#     elif target_duration:
#         max_highlight_duration = target_duration
#         min_segment_duration = 8
#         max_segment_duration = min(30, target_duration / 3)
#     else:
#         # Scale based on video length
#         if video_duration < 300:  # < 5 min videos
#             max_highlight_duration = video_duration * 0.4  # 40% of original
#         elif video_duration < 900:  # 5-15 min videos
#             max_highlight_duration = video_duration * 0.25  # 25% of original
#         else:  # > 15 min videos
#             max_highlight_duration = video_duration * 0.15  # 15% of original
            
#         max_highlight_duration = min(300, max_highlight_duration)  # Cap at 5 minutes
#         min_segment_duration = 10
#         max_segment_duration = 30
    
#     # Calculate how many segments we should have
#     if target_duration:
#         target_num_segments = max(3, int(target_duration / max_segment_duration))
#     else:
#         # Number of segments based on video duration
#         if video_duration < 300:  # < 5 min
#             target_num_segments = 3
#         elif video_duration < 600:  # 5-10 min
#             target_num_segments = 5
#         elif video_duration < 1200:  # 10-20 min
#             target_num_segments = 7
#         else:  # > 20 min
#             target_num_segments = max(7, min(10, int(video_duration / 180)))  # One segment per 3 minutes
    
#     # Create highlight segments
#     highlights = []
    
#     # Always include intro at start
#     intro_duration = min(30, max_segment_duration)
#     highlights.append({
#         "start": 0,
#         "end": intro_duration,
#         "description": "Introduction"
#     })
    
#     # Always include ending/conclusion
#     if video_duration > 60:
#         outro_start = max(0, video_duration - min(30, max_segment_duration))
#         outro_end = video_duration
#         highlights.append({
#             "start": outro_start,
#             "end": outro_end,
#             "description": "Conclusion"
#         })
    
#     # If we have transcript segments, perform advanced analysis
#     if transcript_segments:
#         analyzed_segments = analyze_transcript_advanced(transcript_segments, video_info, video_duration)
        
#         # Calculate how many additional segments we need
#         middle_segments_needed = target_num_segments - len(highlights)
        
#         if middle_segments_needed > 0 and analyzed_segments:
#             # Select highlights balancing importance with coverage
#             middle_highlights = select_highlights_with_coverage(
#                 analyzed_segments, 
#                 video_duration,
#                 middle_segments_needed,
#                 intro_duration,
#                 outro_start
#             )
            
#             highlights.extend(middle_highlights)
#     else:
#         # If no transcript, fall back to evenly spaced segments
#         middle_segments_needed = target_num_segments - len(highlights) 
#         usable_duration = video_duration - intro_duration - (video_duration - outro_start)
        
#         for i in range(middle_segments_needed):
#             # Calculate position
#             position_ratio = (i + 1) / (middle_segments_needed + 1)
#             segment_start = intro_duration + (usable_duration * position_ratio)
            
#             # Add slight jitter for more natural spacing
#             import random
#             jitter = random.uniform(-0.05, 0.05) * max_segment_duration
#             segment_start = max(intro_duration, min(outro_start - max_segment_duration, segment_start + jitter))
            
#             # Create segment
#             highlights.append({
#                 "start": segment_start,
#                 "end": segment_start + max_segment_duration,
#                 "description": f"Segment {i+1}"
#             })
    
#     # Ensure all segments have adequate duration
#     for highlight in highlights:
#         target_length = min(max_segment_duration, max(min_segment_duration, highlight["end"] - highlight["start"]))
#         current_length = highlight["end"] - highlight["start"]
        
#         # If too short, try to extend
#         if current_length < target_length:
#             # How much to add
#             extension_needed = target_length - current_length
            
#             # Try to extend end first (usually better for context)
#             highlight["end"] = min(video_duration, highlight["end"] + extension_needed)
            
#             # If still too short, try to extend beginning
#             current_length = highlight["end"] - highlight["start"]
#             if current_length < target_length:
#                 highlight["start"] = max(0, highlight["start"] - (target_length - current_length))
    
#     # Sort by start time 
#     highlights.sort(key=lambda x: x["start"])
    
#     # Fix any overlap issues
#     for i in range(1, len(highlights)):
#         if highlights[i]["start"] < highlights[i-1]["end"]:
#             highlights[i]["start"] = highlights[i-1]["end"]
#             # Ensure segment isn't too short after adjustment
#             if highlights[i]["end"] - highlights[i]["start"] < min_segment_duration:
#                 # If possible, extend the end time
#                 if i < len(highlights) - 1 and highlights[i]["end"] + min_segment_duration < highlights[i+1]["start"]:
#                     highlights[i]["end"] = highlights[i]["start"] + min_segment_duration
#                 # Otherwise, set to minimum duration but check we don't exceed video length
#                 else:
#                     highlights[i]["end"] = min(video_duration, highlights[i]["start"] + min_segment_duration)
    
#     # Convert transcript segments for snapping
#     if transcript_segments:
#         transcript_for_snap = []
#         for start, end, text in transcript_segments:
#             transcript_for_snap.append({"start": start, "end": end, "text": text})
        
#         # Snap to transcript boundaries for more natural cuts
#         highlights = snap_highlights_to_transcript_boundaries(highlights, transcript_for_snap, max_shift=2.0)
    
#     # Check if we're within target duration
#     total_duration = sum(h["end"] - h["start"] for h in highlights)
    
#     # If we're over target duration, trim segments proportionally
#     if target_duration and total_duration > target_duration * 1.1:  # Allow 10% over
#         excess_ratio = target_duration / total_duration
        
#         # Never trim intro or conclusion
#         intro_segment = highlights[0]
#         conclusion_segment = highlights[-1]
#         middle_segments = highlights[1:-1]
        
#         # Calculate how much we need to reduce middle segments
#         middle_duration = sum(h["end"] - h["start"] for h in middle_segments)
#         target_middle_duration = target_duration - (intro_segment["end"] - intro_segment["start"]) - (conclusion_segment["end"] - conclusion_segment["start"])
        
#         if middle_duration > 0 and target_middle_duration > 0:
#             reduction_ratio = target_middle_duration / middle_duration
            
#             # Adjust each middle segment
#             for h in middle_segments:
#                 segment_duration = h["end"] - h["start"]
#                 new_duration = max(min_segment_duration, segment_duration * reduction_ratio)
#                 h["end"] = h["start"] + new_duration
    
#     # Extract and merge clips
#     logger.info("Extracting highlight clips...")
#     clip_paths, highlight_info = extract_highlights(video_path, highlights)
    
#     logger.info("Merging clips into final video...")
#     output_path = merge_clips(clip_paths, highlight_info, is_reel=is_reel)
    
#     end_time = time.time()
#     logger.info(f"Advanced algorithmic highlights generation completed in {end_time - start_time:.2f} seconds")
    
#     return output_path, highlights

import os
import time
import re
import math
import numpy as np
from collections import Counter
from itertools import groupby
from logger_config import logger
from utils import format_time_duration, format_timestamp
from highlights import extract_highlights, merge_clips, snap_highlights_to_transcript_boundaries

def convert_timestamp_to_seconds(timestamp):
    """Convert HH:MM:SS timestamp to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    return 0

def get_video_type_from_metadata(video_info):
    """Determine the likely video type based on title and description."""
    title = video_info.get('title', '').lower()
    desc = video_info.get('description', '').lower()
    
    # Check for various video types
    if any(term in title or term in desc for term in 
           ['lecture', 'lesson', 'tutorial', 'education', 'learn', 'course', 'teach', 'study']):
        return "educational"
    
    elif any(term in title or term in desc for term in 
            ['game', 'gameplay', 'play', 'match', 'tournament', 'sport', 'versus', 'vs']):
        return "gaming_sports"
    
    elif any(term in title or term in desc for term in 
            ['vlog', 'day in', 'travel', 'trip', 'journey', 'experience', 'daily']):
        return "vlog"
    
    elif any(term in title or term in desc for term in 
            ['review', 'unbox', 'product', 'testing', 'compared', 'comparison']):
        return "review"
    
    elif any(term in title or term in desc for term in 
            ['news', 'report', 'update', 'coverage', 'headlines', 'current events']):
        return "news"
            
    elif any(term in title or term in desc for term in 
            ['interview', 'podcast', 'discussion', 'talk', 'conversation']):
        return "interview"
            
    elif any(term in title or term in desc for term in 
            ['comedy', 'funny', 'humor', 'sketch', 'joke', 'prank']):
        return "entertainment"
            
    elif any(term in title or term in desc for term in 
            ['how to', 'diy', 'guide', 'tips', 'tricks', 'tutorial', 'hack']):
        return "how_to"
    
    # Default to general
    return "general"

def extract_key_phrases(text, max_phrases=3):
    """Extract key phrases from text using tf-idf like approach."""
    # Clean and tokenize text
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Stopwords to ignore
    stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 
                'as', 'for', 'on', 'was', 'by', 'be', 'at', 'this', 'are', 'or',
                'an', 'but', 'not', 'you', 'from', 'have', 'we', 'they', 'i', 'so',
                'there', 'what', 'can', 'all', 'which', 'when', 'been', 'would', 
                'just', 'about', 'some', 'will', 'very', 'also', 'like'}
    
    # Remove stopwords
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Extract phrases (run of 2-3 consecutive meaningful words)
    phrases = []
    for i in range(len(filtered_words) - 1):
        if i < len(filtered_words) - 2:
            trigram = ' '.join(filtered_words[i:i+3])
            phrases.append(trigram)
        
        bigram = ' '.join(filtered_words[i:i+2])
        phrases.append(bigram)
    
    # Score phrases based on word importance
    phrase_scores = {}
    for phrase in phrases:
        if phrase not in phrase_scores:
            words_in_phrase = phrase.split()
            # Score is sum of word frequencies
            score = sum(word_counts[w] for w in words_in_phrase)
            # Multiply by length for slight bonus to longer phrases
            score *= math.sqrt(len(words_in_phrase))
            phrase_scores[phrase] = score
    
    # Get top phrases
    top_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, score in top_phrases[:max_phrases]]

def find_topic_transitions(transcript_segments, min_distance=30):
    """Identify potential topic transitions in the transcript."""
    transitions = []
    segment_texts = []
    segment_starts = []
    
    # Extract all text segments and their start times
    for start, _, text in transcript_segments:
        segment_texts.append(text)
        segment_starts.append(start)
    
    # Need at least several segments to find transitions
    if len(segment_texts) < 10:
        return transitions
    
    # Create simplified bag-of-words for each segment
    bow_segments = []
    for text in segment_texts:
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Remove stopwords
        stopwords = {'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Create bag of words
        bow_segments.append(Counter(words))
    
    # Detect topic shifts by comparing adjacent windows
    window_size = 3  # Compare blocks of segments
    for i in range(window_size, len(bow_segments) - window_size):
        # Previous window
        prev_window = Counter()
        for j in range(i - window_size, i):
            prev_window.update(bow_segments[j])
            
        # Next window
        next_window = Counter()
        for j in range(i, i + window_size):
            next_window.update(bow_segments[j])
            
        # Calculate similarity
        common_words = set(prev_window.keys()) & set(next_window.keys())
        if not common_words:
            similarity = 0
        else:
            # Cosine similarity-inspired measure
            prev_sum = sum(prev_window[w]**2 for w in prev_window)
            next_sum = sum(next_window[w]**2 for w in next_window)
            common_sum = sum(prev_window[w] * next_window[w] for w in common_words)
            
            similarity = common_sum / (math.sqrt(prev_sum) * math.sqrt(next_sum))
        
        # Low similarity indicates topic change
        if similarity < 0.3:  # Threshold for topic change
            transitions.append((segment_starts[i], segment_texts[i], similarity))
    
    # Filter transitions that are too close to each other
    filtered_transitions = []
    for i, (start, text, sim) in enumerate(transitions):
        # Check if this transition is far enough from previous ones
        if i == 0 or start - transitions[i-1][0] >= min_distance:
            filtered_transitions.append((start, text, sim))
    
    return filtered_transitions

def detect_energy_changes(transcript_segments, video_duration):
    """Detect segments with high energy based on speech patterns."""
    energy_points = []
    
    for i, (start, end, text) in enumerate(transcript_segments):
        segment_duration = end - start
        word_count = len(text.split())
        
        # Skip very short or empty segments
        if segment_duration < 1 or word_count < 2:
            continue
        
        energy = 0
        
        # Speech rate
        words_per_second = word_count / segment_duration
        if words_per_second > 3:  # Fast speech
            energy += 5
        
        # Punctuation as energy indicator
        exclamations = text.count('!')
        questions = text.count('?')
        energy += exclamations * 3
        energy += questions * 2
        
        # Special phrases indicating emphasis
        emphasis_phrases = ["important", "key point", "remember", "note that", 
                           "critical", "crucial", "significant", "essential"]
        for phrase in emphasis_phrases:
            if phrase in text.lower():
                energy += 10
                
        # All caps words indicate emphasis
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        energy += len(caps_words) * 3
        
        # Check for numeric facts (numbers often indicate important data points)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        energy += len(numbers) * 2
        
        # If this segment has significant energy, add it to our points
        if energy > 5:
            energy_points.append({
                "start": start,
                "end": end,
                "text": text,
                "energy": energy
            })
    
    # Sort by energy level
    energy_points.sort(key=lambda x: x["energy"], reverse=True)
    return energy_points

def analyze_transcript_advanced(transcript_segments, video_info, video_duration):
    """Advanced transcript analysis combining multiple techniques."""
    segments = []
    
    # Convert transcript segments to more usable format
    for start, end, text in transcript_segments:
        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "importance": 0  # Will be calculated
        })
    
    # Get video type
    video_type = get_video_type_from_metadata(video_info)
    
    # Find topic transitions
    transitions = find_topic_transitions(transcript_segments)
    transition_times = [t[0] for t in transitions]
    
    # Find energy points
    energy_points = detect_energy_changes(transcript_segments, video_duration)
    energy_times = [p["start"] for p in energy_points]
    
    # Calculate overall importance score for each segment
    for i, segment in enumerate(segments):
        importance = 0
        text = segment["text"].lower()
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Position-based importance (intro/conclusion)
        if start_time <= video_duration * 0.05:  # First 5%
            importance += 30
        elif start_time >= video_duration * 0.9:  # Last 10%
            importance += 25
        
        # Video-type specific weights
        if video_type == "educational":
            # In educational videos, explanations, definitions, examples are important
            if any(term in text for term in ["example", "definition", "means", "is called", 
                                           "defined as", "this is", "for instance"]):
                importance += 20
                
            # Numerical information is often important in educational content
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
            importance += len(numbers) * 5
            
        elif video_type == "gaming_sports":
            # In gaming/sports, exciting moments and results are important
            if any(term in text for term in ["win", "score", "point", "goal", "shot", 
                                           "amazing", "incredible", "victory"]):
                importance += 25
                
        elif video_type == "vlog":
            # In vlogs, transitions between locations and activities are important
            if any(term in text for term in ["now we're", "here we are", "next", "then",
                                           "after that", "arrived", "going to"]):
                importance += 15
                
        elif video_type == "review":
            # In reviews, ratings, pros/cons, and conclusions are important
            if any(term in text for term in ["rating", "recommend", "score", "stars",
                                           "pros", "cons", "verdict", "conclusion"]):
                importance += 20
                
        elif video_type == "how_to":
            # In how-to videos, steps and warnings are important
            if any(term in text for term in ["step", "next", "first", "then", "finally",
                                           "warning", "caution", "important to note"]):
                importance += 20
        
        # Topic transition bonus (segment starts near a detected topic change)
        for trans_time in transition_times:
            if abs(start_time - trans_time) < 5:  # Within 5 seconds
                importance += 30
                break
        
        # Energy level bonus
        for energy_time in energy_times:
            if abs(start_time - energy_time) < 3:  # Within 3 seconds
                # Find the energy point and add its energy score
                for point in energy_points:
                    if point["start"] == energy_time:
                        importance += point["energy"]
                        break
                break
        
        # Extract key phrases for semantic understanding
        key_phrases = extract_key_phrases(text)
        importance += len(key_phrases) * 5
        
        # Content-based importance
        emphasis_terms = ["important", "key", "critical", "essential", "vital", 
                         "remember", "note that", "crucial", "significant", "main"]
        conclusion_terms = ["in conclusion", "to summarize", "in summary", "finally", 
                           "to conclude", "wrapping up", "to finish", "lastly"]
        highlight_terms = ["highlight", "takeaway", "ultimately", "point", "focus"]
        
        for term in emphasis_terms:
            if term in text:
                importance += 20
        
        for term in conclusion_terms:
            if term in text:
                importance += 15
        
        for term in highlight_terms:
            if term in text:
                importance += 10
                
        # Duration bonus - slightly favor longer segments that have complete thoughts
        segment_duration = end_time - start_time
        ideal_duration = 20  # Around 20 seconds is often good for a highlight segment
        duration_factor = 1 - abs(segment_duration - ideal_duration) / ideal_duration
        duration_factor = max(0, duration_factor)  # Keep it non-negative
        importance += duration_factor * 10
        
        # Store calculated importance
        segments[i]["importance"] = importance
    
    # Sort segments by importance
    segments.sort(key=lambda x: x["importance"], reverse=True)
    
    return segments

def select_highlights_with_coverage(important_segments, video_duration, target_num_segments, 
                                  intro_duration, outro_start):
    """Select highlights balancing importance with good video coverage."""
    highlights = []
    
    # Create time buckets for even distribution
    num_buckets = max(3, target_num_segments-2)  # Excluding intro and conclusion
    bucket_duration = (outro_start - intro_duration) / num_buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Assign segments to buckets
    for segment in important_segments:
        if segment["start"] < intro_duration or segment["start"] >= outro_start:
            continue  # Skip segments in intro or conclusion range
            
        bucket_index = min(num_buckets-1, int((segment["start"] - intro_duration) / bucket_duration))
        buckets[bucket_index].append(segment)
    
    # Sort segments within each bucket by importance
    for bucket in buckets:
        bucket.sort(key=lambda x: x["importance"], reverse=True)
    
    # Take the most important segment from each bucket
    segments_per_bucket = 1
    segments_to_select = min(target_num_segments-2, sum(len(b) > 0 for b in buckets))
    
    # If we have more buckets than segments to select, prioritize the buckets with 
    # the most important segments
    if segments_to_select < len(buckets):
        # Get the most important segment from each bucket
        best_from_each = []
        for i, bucket in enumerate(buckets):
            if bucket:
                best_from_each.append((i, bucket[0]["importance"]))
        
        # Sort buckets by the importance of their best segment
        best_from_each.sort(key=lambda x: x[1], reverse=True)
        selected_bucket_indices = [idx for idx, _ in best_from_each[:segments_to_select]]
    else:
        selected_bucket_indices = range(len(buckets))
    
    # Select segments from chosen buckets
    for bucket_idx in selected_bucket_indices:
        bucket = buckets[bucket_idx]
        if not bucket:
            continue
            
        # Take up to segments_per_bucket from this bucket
        for i in range(min(segments_per_bucket, len(bucket))):
            segment = bucket[i]
            highlights.append({
                "start": segment["start"],
                "end": segment["end"],
                "description": f"Key moment: {segment['text'][:50]}..." if len(segment['text']) > 50 else f"Key moment: {segment['text']}"
            })
    
    return highlights

def merge_transcript_segments_into_complete_sentences(transcript_segments):
    """
    Merge transcript segments to form complete sentences.
    
    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        
    Returns:
        List of (start_time, end_time, text) tuples with merged segments for complete sentences
    """
    if not transcript_segments:
        return []
    
    import re
    
    # Define sentence terminators
    sentence_terminators = re.compile(r'[.!?]')
    
    merged_segments = []
    current_start = transcript_segments[0][0]
    current_text = ""
    
    for i, (start, end, text) in enumerate(transcript_segments):
        # Add current segment text
        current_text += text
        
        # Check if we have a complete sentence
        match = sentence_terminators.search(current_text)
        
        # If we have a complete sentence OR this is the last segment
        if match or i == len(transcript_segments) - 1:
            # If we found a sentence terminator
            if match:
                # Find the last sentence terminator in the text
                last_terminator_idx = max(i for i, c in enumerate(current_text) if c in '.!?')
                
                # Split into the complete sentence and the remainder
                complete_sentence = current_text[:last_terminator_idx + 1].strip()
                remainder = current_text[last_terminator_idx + 1:].strip()
                
                # Add the complete sentence to our merged segments
                merged_segments.append((current_start, end, complete_sentence))
                
                # If there's a remainder, start a new segment with it
                if remainder:
                    current_start = end  # Start time is the end time of the current segment
                    current_text = remainder
                else:
                    current_text = ""
                    
                    # If this is not the last segment, update current_start for the next segment
                    if i < len(transcript_segments) - 1:
                        current_start = transcript_segments[i + 1][0]
            else:
                # If this is the last segment and doesn't end with a terminator, add it anyway
                merged_segments.append((current_start, end, current_text.strip()))
                current_text = ""
        
    return merged_segments

def analyze_complete_sentences(merged_segments, video_info, video_duration):
    """
    Analyze the importance of complete sentence segments for highlight generation.
    
    Builds upon the analyze_transcript_advanced function but works with merged segments.
    
    Args:
        merged_segments: List of (start_time, end_time, text) tuples with complete sentences
        video_info: Dictionary with video title and description
        video_duration: Duration of the video in seconds
        
    Returns:
        List of segments sorted by importance
    """
    segments = []
    
    # Convert merged segments to more usable format
    for start, end, text in merged_segments:
        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "importance": 0  # Will be calculated
        })
    
    # Get video type
    video_type = get_video_type_from_metadata(video_info)
    
    # Find topic transitions using the merged segments
    transitions = find_topic_transitions(merged_segments)
    transition_times = [t[0] for t in transitions]
    
    # Find energy points
    energy_points = detect_energy_changes(merged_segments, video_duration)
    energy_times = [p["start"] for p in energy_points]
    
    # Calculate overall importance score for each segment
    for i, segment in enumerate(segments):
        importance = 0
        text = segment["text"].lower()
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Position-based importance (intro/conclusion)
        if start_time <= video_duration * 0.05:  # First 5%
            importance += 30
        elif start_time >= video_duration * 0.9:  # Last 10%
            importance += 25
        
        # Video-type specific weights
        if video_type == "educational":
            # In educational videos, explanations, definitions, examples are important
            if any(term in text for term in ["example", "definition", "means", "is called", 
                                           "defined as", "this is", "for instance"]):
                importance += 20
                
            # Numerical information is often important in educational content
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
            importance += len(numbers) * 5
            
        elif video_type == "gaming_sports":
            # In gaming/sports, exciting moments and results are important
            if any(term in text for term in ["win", "score", "point", "goal", "shot", 
                                           "amazing", "incredible", "victory"]):
                importance += 25
                
        elif video_type == "vlog":
            # In vlogs, transitions between locations and activities are important
            if any(term in text for term in ["now we're", "here we are", "next", "then",
                                           "after that", "arrived", "going to"]):
                importance += 15
                
        elif video_type == "review":
            # In reviews, ratings, pros/cons, and conclusions are important
            if any(term in text for term in ["rating", "recommend", "score", "stars",
                                           "pros", "cons", "verdict", "conclusion"]):
                importance += 20
                
        elif video_type == "how_to":
            # In how-to videos, steps and warnings are important
            if any(term in text for term in ["step", "next", "first", "then", "finally",
                                           "warning", "caution", "important to note"]):
                importance += 20
        
        # Topic transition bonus (segment starts near a detected topic change)
        for trans_time in transition_times:
            if abs(start_time - trans_time) < 5:  # Within 5 seconds
                importance += 30
                break
        
        # Energy level bonus
        for energy_time in energy_times:
            if abs(start_time - energy_time) < 3:  # Within 3 seconds
                # Find the energy point and add its energy score
                for point in energy_points:
                    if point["start"] == energy_time:
                        importance += point["energy"]
                        break
                break
        
        # Extract key phrases for semantic understanding
        key_phrases = extract_key_phrases(text)
        importance += len(key_phrases) * 5
        
        # Content-based importance
        emphasis_terms = ["important", "key", "critical", "essential", "vital", 
                         "remember", "note that", "crucial", "significant", "main"]
        conclusion_terms = ["in conclusion", "to summarize", "in summary", "finally", 
                           "to conclude", "wrapping up", "to finish", "lastly"]
        highlight_terms = ["highlight", "takeaway", "ultimately", "point", "focus"]
        
        for term in emphasis_terms:
            if term in text:
                importance += 20
        
        for term in conclusion_terms:
            if term in text:
                importance += 15
        
        for term in highlight_terms:
            if term in text:
                importance += 10
                
        # Sentence completeness bonus - giving more weight to complete sentences
        if text.strip().endswith(('.', '!', '?')):
            importance += 10
            
        # Length bonus for substantial sentences
        word_count = len(text.split())
        if word_count > 10:  # Longer sentences often contain more content
            importance += min(10, word_count / 5)  # Cap at +10
            
        # Duration bonus - slightly favor longer segments that have complete thoughts
        segment_duration = end_time - start_time
        ideal_duration = 20  # Around 20 seconds is often good for a highlight segment
        duration_factor = 1 - min(1, abs(segment_duration - ideal_duration) / ideal_duration)
        importance += duration_factor * 10
        
        # Store calculated importance
        segments[i]["importance"] = importance
    
    # Sort segments by importance
    segments.sort(key=lambda x: x["importance"], reverse=True)
    
    return segments

async def generate_highlights_algorithmically(video_path, transcript_segments, video_info, 
                                             target_duration=None, is_reel=False):
    """
    Generate highlights using advanced algorithmic analysis without LLM.
    
    Args:
        video_path: Path to the video file
        transcript_segments: List of transcript segments with timestamps
        video_info: Dictionary with video title and description
        target_duration: Target duration for highlights in seconds
        is_reel: Whether to optimize for short-form content
        
    Returns:
        Tuple of (output_path, highlight_segments)
    """
    start_time = time.time()
    print(f"Generating {'reel' if is_reel else 'highlights'} using advanced analysis...")
    
    # Get video duration
    from moviepy import VideoFileClip
    with VideoFileClip(video_path) as video:
        video_duration = video.duration
    
    # Calculate target highlight parameters
    if is_reel:
        max_highlight_duration = 60  # 1 minute max for reels
        min_segment_duration = 5     # 5 seconds minimum per segment
        max_segment_duration = 15    # 15 seconds maximum per segment
    elif target_duration:
        max_highlight_duration = target_duration
        min_segment_duration = 8
        max_segment_duration = min(30, target_duration / 3)
    else:
        # Scale based on video length
        if video_duration < 300:  # < 5 min videos
            max_highlight_duration = video_duration * 0.4  # 40% of original
        elif video_duration < 900:  # 5-15 min videos
            max_highlight_duration = video_duration * 0.25  # 25% of original
        else:  # > 15 min videos
            max_highlight_duration = video_duration * 0.15  # 15% of original
            
        max_highlight_duration = min(300, max_highlight_duration)  # Cap at 5 minutes
        min_segment_duration = 10
        max_segment_duration = 30
    
    # Calculate how many segments we should have
    if target_duration:
        target_num_segments = max(3, int(target_duration / max_segment_duration))
    else:
        # Number of segments based on video duration
        if video_duration < 300:  # < 5 min
            target_num_segments = 3
        elif video_duration < 600:  # 5-10 min
            target_num_segments = 5
        elif video_duration < 1200:  # 10-20 min
            target_num_segments = 7
        else:  # > 20 min
            target_num_segments = max(7, min(10, int(video_duration / 180)))  # One segment per 3 minutes
    
    # Create highlight segments
    highlights = []
    
    # Always include intro at start
    intro_duration = min(30, max_segment_duration)
    highlights.append({
        "start": 0,
        "end": intro_duration,
        "description": "Introduction"
    })
    
    # Always include ending/conclusion
    if video_duration > 60:
        outro_start = max(0, video_duration - min(30, max_segment_duration))
        outro_end = video_duration
        highlights.append({
            "start": outro_start,
            "end": outro_end,
            "description": "Conclusion"
        })
    
    # If we have transcript segments, perform advanced analysis
    if transcript_segments:
        # Merge transcript segments into complete sentences first
        merged_transcript_segments = merge_transcript_segments_into_complete_sentences(transcript_segments)
        
        # Analyze merged segments
        analyzed_segments = analyze_complete_sentences(merged_transcript_segments, video_info, video_duration)
        
        # Calculate how many additional segments we need
        middle_segments_needed = target_num_segments - len(highlights)
        
        if middle_segments_needed > 0 and analyzed_segments:
            # Select highlights balancing importance with coverage
            middle_highlights = select_highlights_with_coverage(
                analyzed_segments, 
                video_duration,
                middle_segments_needed,
                intro_duration,
                outro_start
            )
            
            highlights.extend(middle_highlights)
    else:
        # If no transcript, fall back to evenly spaced segments
        middle_segments_needed = target_num_segments - len(highlights) 
        usable_duration = video_duration - intro_duration - (video_duration - outro_start)
        
        for i in range(middle_segments_needed):
            # Calculate position
            position_ratio = (i + 1) / (middle_segments_needed + 1)
            segment_start = intro_duration + (usable_duration * position_ratio)
            
            # Add slight jitter for more natural spacing
            import random
            jitter = random.uniform(-0.05, 0.05) * max_segment_duration
            segment_start = max(intro_duration, min(outro_start - max_segment_duration, segment_start + jitter))
            
            # Create segment
            highlights.append({
                "start": segment_start,
                "end": segment_start + max_segment_duration,
                "description": f"Segment {i+1}"
            })
    
    # Ensure all segments have adequate duration
    for highlight in highlights:
        target_length = min(max_segment_duration, max(min_segment_duration, highlight["end"] - highlight["start"]))
        current_length = highlight["end"] - highlight["start"]
        
        # If too short, try to extend
        if current_length < target_length:
            # How much to add
            extension_needed = target_length - current_length
            
            # Try to extend end first (usually better for context)
            highlight["end"] = min(video_duration, highlight["end"] + extension_needed)
            
            # If still too short, try to extend beginning
            current_length = highlight["end"] - highlight["start"]
            if current_length < target_length:
                highlight["start"] = max(0, highlight["start"] - (target_length - current_length))
    
    # Sort by start time 
    highlights.sort(key=lambda x: x["start"])
    
    # Fix any overlap issues
    for i in range(1, len(highlights)):
        if highlights[i]["start"] < highlights[i-1]["end"]:
            highlights[i]["start"] = highlights[i-1]["end"]
            # Ensure segment isn't too short after adjustment
            if highlights[i]["end"] - highlights[i]["start"] < min_segment_duration:
                # If possible, extend the end time
                if i < len(highlights) - 1 and highlights[i]["end"] + min_segment_duration < highlights[i+1]["start"]:
                    highlights[i]["end"] = highlights[i]["start"] + min_segment_duration
                # Otherwise, set to minimum duration but check we don't exceed video length
                else:
                    highlights[i]["end"] = min(video_duration, highlights[i]["start"] + min_segment_duration)
    
    # Convert merged transcript segments for snapping
    if transcript_segments:
        # Use the merged transcript segments for snapping
        merged_transcript_for_snap = []
        for start, end, text in merge_transcript_segments_into_complete_sentences(transcript_segments):
            merged_transcript_for_snap.append({"start": start, "end": end, "text": text})
        
        # Snap to transcript boundaries for more natural cuts
        highlights = snap_highlights_to_transcript_boundaries(highlights, merged_transcript_for_snap, max_shift=2.0)
    
    # Check if we're within target duration
    total_duration = sum(h["end"] - h["start"] for h in highlights)
    
    # If we're over target duration, trim segments proportionally
    if target_duration and total_duration > target_duration * 1.1:  # Allow 10% over
        excess_ratio = target_duration / total_duration
        
        # Never trim intro or conclusion
        intro_segment = highlights[0]
        conclusion_segment = highlights[-1]
        middle_segments = highlights[1:-1]
        
        # Calculate how much we need to reduce middle segments
        middle_duration = sum(h["end"] - h["start"] for h in middle_segments)
        target_middle_duration = target_duration - (intro_segment["end"] - intro_segment["start"]) - (conclusion_segment["end"] - conclusion_segment["start"])
        
        if middle_duration > 0 and target_middle_duration > 0:
            reduction_ratio = target_middle_duration / middle_duration
            
            # Adjust each middle segment
            for h in middle_segments:
                segment_duration = h["end"] - h["start"]
                new_duration = max(min_segment_duration, segment_duration * reduction_ratio)
                h["end"] = h["start"] + new_duration
    
    # Extract and merge clips
    logger.info("Extracting highlight clips...")
    clip_paths, highlight_info = extract_highlights(video_path, highlights)
    
    logger.info("Merging clips into final video...")
    output_path = merge_clips(clip_paths, highlight_info, is_reel=is_reel)
    
    end_time = time.time()
    logger.info(f"Advanced algorithmic highlights generation completed in {end_time - start_time:.2f} seconds")
    
    return output_path, highlights

def parse_transcript_with_timestamps(transcript_text):
    """
    Parse transcript text with timestamps in the format:
    00:00:00 - 00:00:03: I dedicated the past two years to understanding
    
    Args:
        transcript_text: String containing transcript with timestamps
        
    Returns:
        List of (start_time, end_time, text) tuples
    """
    import re
    
    # Regex to match timestamp and text
    pattern = r'(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}): (.*?)(?=\n\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}:|$)'
    
    # Alternative pattern for MM:SS format
    alt_pattern = r'(\d{2}:\d{2}) - (\d{2}:\d{2}): (.*?)(?=\n\d{2}:\d{2} - \d{2}:\d{2}:|$)'
    
    # Find all matches
    matches = re.findall(pattern, transcript_text, re.DOTALL)
    
    # If no matches with HH:MM:SS format, try MM:SS format
    if not matches:
        matches = re.findall(alt_pattern, transcript_text, re.DOTALL)
    
    # Convert to proper format
    segments = []
    for start_time, end_time, text in matches:
        start_seconds = convert_timestamp_to_seconds(start_time)
        end_seconds = convert_timestamp_to_seconds(end_time)
        segments.append((start_seconds, end_seconds, text.strip()))
    
    return segments