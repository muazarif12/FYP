from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import time
from constants import OUTPUT_DIR
from logger_config import logger
from utils import format_time_duration, format_timestamp
import ollama
from moviepy import VideoFileClip  # Ensure correct import
import os
import time
import re
import ollama
from functools import wraps
from moviepy import VideoFileClip, concatenate_videoclips, VideoClip
from moviepy.video.fx import CrossFadeIn
import random
import json
import shutil
from concurrent.futures import ThreadPoolExecutor


def snap_highlights_to_transcript_boundaries(highlights, transcript, max_shift=2.0):
    """Refine highlight start/end times to align with transcript boundaries."""
    if not transcript:
        return highlights

    # Gather all boundaries from transcript
    boundaries = set()
    for seg in transcript:
        boundaries.add(seg["start"])
        boundaries.add(seg["end"])
    boundary_list = sorted(boundaries)

    def find_nearest_boundary(time):
        """Find nearest boundary within max_shift."""
        best = None
        best_diff = float('inf')
        for b in boundary_list:
            diff = abs(b - time)
            if diff < best_diff:
                best_diff = diff
                best = b
            else:
                if b > time and diff > best_diff:
                    break
        return (best, best_diff)

    snapped = []
    for hl in highlights:
        start_t = hl["start"]
        end_t = hl["end"]
        # Snap start
        best_boundary, diff = find_nearest_boundary(start_t)
        if diff <= max_shift:
            start_t = best_boundary
        # Snap end
        best_boundary, diff = find_nearest_boundary(end_t)
        if diff <= max_shift:
            end_t = best_boundary
        # Ensure valid
        if end_t > start_t:
            snapped.append({
                "start": start_t,
                "end": end_t,
                "description": hl["description"]
            })
    # Sort again
    snapped.sort(key=lambda x: x["start"])
    return snapped



def analyze_transcript_importance(transcript, video_duration):
    """Pre-analyze transcript to identify potentially important moments."""
    importance_markers = {}
    
    # Identify potential markers of importance
    for i, seg in enumerate(transcript):
        text = seg["text"].lower()
        start = seg["start"]
        end = seg["end"]
        importance = 0
        
        # Position-based importance (intro/conclusion are important)
        if start <= video_duration * 0.05:  # First 5%
            importance += 30
        elif start >= video_duration * 0.9:  # Last 10%
            importance += 25
        
        # Content-based importance
        emphasis_terms = ["important", "key", "critical", "essential", "vital", 
                         "remember", "note that", "crucial", "significant"]
        conclusion_terms = ["in conclusion", "to summarize", "in summary", "finally", 
                           "to conclude", "wrapping up", "to finish"]
        highlight_terms = ["highlight", "main point", "takeaway", "major", "ultimately"]
        
        for term in emphasis_terms:
            if term in text:
                importance += 20
        
        for term in conclusion_terms:
            if term in text:
                importance += 15
        
        for term in highlight_terms:
            if term in text:
                importance += 10
        
        # Detect questions or statements that might be important
        if "?" in text:
            importance += 5
        if "!" in text:
            importance += 5
            
        # Look for sequence markers
        sequence_markers = ["first", "second", "third", "next", "finally", "then", "lastly"]
        for marker in sequence_markers:
            if marker in text.split():  # Only match whole words
                importance += 8
        
        # More speech pace/energy indicators
        words_per_second = len(text.split()) / max(1, (end - start))
        if words_per_second > 3:  # Fast speech can indicate excitement or important points
            importance += 5
            
        importance_markers[i] = importance
    
    return importance_markers

def select_representative_segments(transcript, importance_markers, max_samples=50):
    """Select representative segments for LLM analysis based on importance."""
    if len(transcript) <= max_samples:
        return transcript
    
    # Always include beginning and end
    start_segments = transcript[:min(10, len(transcript)//10)]
    end_segments = transcript[-min(10, len(transcript)//10):]
    
    # Sort remaining segments by importance
    middle_segments = transcript[len(start_segments):-len(end_segments)]
    segment_importance = [(i+len(start_segments), importance_markers.get(i+len(start_segments), 0)) 
                          for i in range(len(middle_segments))]
    
    # Sort by importance (highest first)
    segment_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Take top segments up to max_samples - len(start) - len(end)
    remaining_slots = max_samples - len(start_segments) - len(end_segments)
    important_indices = [idx for idx, _ in segment_importance[:remaining_slots]]
    important_indices.sort()  # Sort indices to maintain original order
    
    # Get important middle segments
    important_middle = [transcript[idx] for idx in important_indices]
    
    # Combine and return
    return start_segments + important_middle + end_segments

def ensure_opening_and_closing_segments(highlights, transcript, video_duration):
    """Ensure we have opening and closing segments in our highlights."""
    # Check if we have an opening segment
    has_opening = any(h["start"] <= 30 for h in highlights)
    has_closing = any(h["end"] >= video_duration - 30 for h in highlights)
    
    # Add opening if missing
    if not has_opening and transcript:
        early_segments = [seg for seg in transcript if seg["start"] < 60]
        if early_segments:
            # Find a good opening clip (about 15-30 seconds)
            start = 0
            end = min(30, early_segments[-1]["end"])
            highlights.insert(0, {
                "start": start,
                "end": end,
                "description": "Opening segment"
            })
    
    # Add closing if missing
    if not has_closing and transcript:
        late_segments = [seg for seg in transcript if seg["end"] > video_duration - 60]
        if late_segments:
            # Find a good closing clip (about 15-30 seconds)
            start = max(video_duration - 30, late_segments[0]["start"])
            end = video_duration
            highlights.append({
                "start": start,
                "end": end,
                "description": "Closing segment"
            })
    
    # Sort by start time
    highlights.sort(key=lambda x: x["start"])

def validate_highlight_quality(highlights, importance_markers, transcript, video_duration):
    """Final validation to ensure highlight quality."""
    if not highlights:
        return []
    
    # Check for adequate coverage
    coverage_gaps = []
    for i in range(1, len(highlights)):
        gap = highlights[i]["start"] - highlights[i-1]["end"]
        if gap > video_duration * 0.25:  # Large gap (>25% of video)
            midpoint = (highlights[i]["start"] + highlights[i-1]["end"]) / 2
            coverage_gaps.append(midpoint)
    
    # If we have significant gaps, try to fill them with important content
    if coverage_gaps and len(highlights) < 12:  # Don't add too many highlights
        for gap_center in coverage_gaps:
            # Find nearby important segments
            nearby_important = []
            for i, importance in importance_markers.items():
                if i < len(transcript) and importance > 15:  # Only consider fairly important segments
                    seg = transcript[i]
                    distance = abs((seg["start"] + seg["end"])/2 - gap_center)
                    if distance < video_duration * 0.1:  # Within 10% of video length
                        nearby_important.append((seg, importance, distance))
            
            # Add the most important nearby segment
            if nearby_important:
                nearby_important.sort(key=lambda x: (x[1] * 10) - x[2], reverse=True)  # Prioritize importance but consider distance
                best_seg = nearby_important[0][0]
                highlights.append({
                    "start": best_seg["start"],
                    "end": best_seg["end"],
                    "description": f"Important point at {format_timestamp(best_seg['start'])}"
                })
    
    # Final sorting
    highlights.sort(key=lambda x: x["start"])
    
    return highlights

def improve_highlight_prompt(video_info, video_duration, target_num_highlights, segment_length, 
                           min_highlight_duration, max_highlight_duration, is_reel=False, 
                           sample_segments=None, transcript=None):
    """Create an improved prompt for the LLM based on video content type."""
    if is_reel:
        duration_line = "This is for a short social media reel (30-60 seconds total)."
    else:
        duration_line = f"Total highlight duration range: {min_highlight_duration:.1f}-{max_highlight_duration:.1f} seconds."

    video_type_hint = ""
    # Try to detect video type from title/description
    title = video_info.get('title', '').lower()
    desc = video_info.get('description', '').lower()
    
    # Detect video type
    if any(term in title or term in desc for term in ['game', 'match', 'tournament', 'championship', 'sport', 'team', 'play', 'goal', 'score']):
        video_type_hint = """
        This appears to be a SPORTS video. Focus on:
        - Key plays, goals, or points scored
        - Dramatic turning points in the game or match
        - Celebrations or notable reactions
        - Introductions of key players and conclusion
        """
    elif any(term in title or term in desc for term in ['lecture', 'lesson', 'tutorial', 'educational', 'learn', 'course', 'teach', 'study', 'academic']):
        video_type_hint = """
        This appears to be an EDUCATIONAL video. Focus on:
        - Main thesis or key learning objectives stated at beginning
        - Core concepts and crucial explanations
        - Examples that illustrate difficult concepts
        - Summary points and conclusions
        - Key formulas, definitions, or frameworks
        """
    elif any(term in title or term in desc for term in ['review', 'unbox', 'product', 'test', 'versus', 'comparison']):
        video_type_hint = """
        This appears to be a REVIEW or PRODUCT video. Focus on:
        - Initial introduction of what's being reviewed
        - Key features highlighted
        - Pros and cons mentioned
        - Surprising revelations or unique observations
        - Final verdict or recommendation
        """
    elif any(term in title or term in desc for term in ['vlog', 'day', 'life', 'travel', 'experience', 'trip', 'journey']):
        video_type_hint = """
        This appears to be a VLOG or EXPERIENTIAL video. Focus on:
        - Setting the scene/introduction to the experience
        - Most visually interesting moments
        - Emotional high points or reactions
        - Cultural or unique experiences highlighted
        - Reflections or conclusions
        """
    
    # Handle potential None values for transcript and sample_segments
    transcript_length = len(transcript) if transcript is not None else 0
    if sample_segments is None:
        sample_segments = []
    
    # Important fix: Escape the curly braces in the example JSON by doubling them
    # This prevents them from being interpreted as format specifiers
    prompt = f"""
    You are an expert video editor specializing in creating perfect highlight reels. Your task is to analyze this transcript and identify the most important, engaging, and representative moments to include in a highlight reel.

    VIDEO DETAILS:
    - Title: "{video_info.get('title', 'Unknown')}"
    - Duration: {format_time_duration(video_duration)} ({video_duration:.1f} s)
    - Description: "{(video_info.get('description') or '')[:300]}..."
    
    {video_type_hint}

    GOAL:
    Create a professional highlight reel that captures ALL key moments and maintains the narrative integrity of the original video while being much shorter.
    
    HIGHLIGHT CRITERIA:
    1. Select exactly {target_num_highlights} highlights.
    2. Each highlight should be approximately {segment_length:.1f} seconds long.
    3. {duration_line}
    4. ALWAYS include an introduction segment from the beginning of the video to establish context.
    5. ALWAYS include a conclusion segment from the end of the video to maintain narrative closure.
    6. Focus on identifying:
       - Key statements or revelations
       - Emotionally charged moments
       - Surprises or unexpected elements
       - Critical information or main points
       - Visual or auditory peaks (indicated by excitement in speech)
    7. Ensure highlights are distributed throughout the video, avoiding clusters.
    8. Prioritize segments that are self-contained and make sense without additional context.

    TRANSCRIPT SAMPLE (not entire transcript):
    {json.dumps(sample_segments, indent=2)}

    (N.B. The full transcript has {transcript_length} segments.)

    INSTRUCTIONS:
    - Analyze the transcript thoroughly, looking for moments of significance.
    - For each segment, assess its importance to the overall narrative.
    - Return ONLY a strict JSON array of highlight objects, no extra text.

    FINAL RESPONSE EXAMPLE:
    [
      {{
        "start": 10.0,
        "end": 25.0,
        "description": "Host explains a key concept with excitement."
      }},
      ...
    ]
    """
    return prompt

async def analyze_transcript_for_highlights(transcript, video_info, video_duration, target_total_duration=None, is_reel=False):
    """Use Ollama to find highlight segments, then snap them to speaker boundaries."""
    start_time = time.time()

    # Decide highlight durations with better scaling
    if is_reel:
        # For reels, aim for very short clips (5-15 seconds each)
        target_num_highlights = 4
        segment_length = 7.5  # seconds per clip
        min_highlight_duration = 30  # total seconds
        max_highlight_duration = 60  # total seconds
    elif target_total_duration is not None:
        min_highlight_duration = target_total_duration
        max_highlight_duration = target_total_duration
        
        # More intelligent calculation based on target duration
        if target_total_duration < 90:
            target_num_highlights = 3
        elif target_total_duration < 180:
            target_num_highlights = 4
        elif target_total_duration < 300:
            target_num_highlights = 5
        else:
            target_num_highlights = max(5, min(10, int(target_total_duration / 60)))
            
        segment_length = target_total_duration / target_num_highlights
    else:
        # More sophisticated scaling based on video duration
        if video_duration < 120:  # < 2 min videos
            min_pct, max_pct = 40, 60
            target_num_highlights = 3
        elif video_duration < 300:  # 2-5 min videos
            min_pct, max_pct = 25, 40
            target_num_highlights = 4
        elif video_duration < 600:  # 5-10 min videos
            min_pct, max_pct = 15, 30
            target_num_highlights = 5
        elif video_duration < 1200:  # 10-20 min videos
            min_pct, max_pct = 10, 25
            target_num_highlights = 6
        elif video_duration < 1800:  # 20-30 min videos
            min_pct, max_pct = 8, 20
            target_num_highlights = 7
        else:  # > 30 min videos
            min_pct, max_pct = 5, 15
            target_num_highlights = max(8, min(12, int(video_duration / 300)))
            
        min_highlight_duration = (video_duration * min_pct) / 100
        max_highlight_duration = (video_duration * max_pct) / 100
        segment_length = max_highlight_duration / target_num_highlights
        segment_length = min(max(15, segment_length), 90)
    
    # Pre-analyze transcript for important markers - check for None
    if transcript is None:
        logger.warning("Transcript is None. Using fallback.")
        return create_default_highlights(video_duration, target_num_highlights, segment_length, is_reel)
        
    importance_markers = analyze_transcript_importance(transcript, video_duration)
    
    # Build sample segments more intelligently
    sample_segments = select_representative_segments(transcript, importance_markers)
    
    # Create improved prompt
    prompt = improve_highlight_prompt(
        video_info, 
        video_duration,
        target_num_highlights,
        segment_length,
        min_highlight_duration,
        max_highlight_duration,
        is_reel,
        sample_segments,
        transcript  # Add this parameter - this was missing!
    )

    try:
        logger.info("Sending transcript to Ollama LLM for highlight analysis...")
        response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "system", "content": prompt}])
        logger.info("Response received from LLM model.")
        raw_content = response["message"]["content"]

        # Try multiple approaches to extract valid JSON
        highlights = None

        # First try direct JSON parsing
        try:
            highlights = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed, trying regex extraction...")

            # Try regex extraction
            json_match = re.search(r'\[\s*{.*}\s*\]', raw_content.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    highlights_str = json_match.group(0)
                    highlights = json.loads(highlights_str)
                except json.JSONDecodeError:
                    logger.warning("Regex JSON extraction failed")

            # If still no valid JSON, try to fix common issues
            if not highlights:
                logger.info("Trying to fix and extract JSON...")
                # Remove markdown code blocks
                cleaned = re.sub(r'```json\s*|\s*```', '', raw_content)
                # Try to find array bounds
                start_idx = cleaned.find('[')
                end_idx = cleaned.rfind(']')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    try:
                        json_str = cleaned[start_idx:end_idx+1]
                        highlights = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Fixed JSON extraction failed")

        # Validate
        if not highlights or not isinstance(highlights, list) or len(highlights) == 0:
            logger.warning("Empty or invalid highlights from LLM. Using fallback.")
            return create_default_highlights(video_duration, target_num_highlights, segment_length, is_reel)

        # Clean up & merge
        cleaned = []
        total_hl_duration = 0
        for h in highlights:
            if not all(k in h for k in ("start", "end", "description")):
                continue
            s, e = float(h["start"]), float(h["end"])
            s = max(0, s)
            e = min(video_duration, e)
            if e <= s or (e - s) < 2:
                continue
            if (e - s) > 180:
                e = s + 180
            cleaned.append({
                "start": s,
                "end": e,
                "description": h["description"].strip()
            })
            total_hl_duration += (e - s)

        if total_hl_duration < (min_highlight_duration * 0.5):
            logger.warning("Total highlight duration too short. Switching to fallback.")
            cleaned = create_default_highlights(video_duration, target_num_highlights, segment_length, is_reel)

        cleaned.sort(key=lambda x: x["start"])
        merged = []
        for c in cleaned:
            if not merged or c["start"] > merged[-1]["end"]:
                merged.append(c)
            else:
                merged[-1]["end"] = max(merged[-1]["end"], c["end"])
                merged[-1]["description"] += " + " + c["description"]

        # Ensure we always have opening and closing segments 
        ensure_opening_and_closing_segments(merged, transcript, video_duration)
        
        # Snap to transcript boundaries for natural cut points
        snapped = snap_highlights_to_transcript_boundaries(merged, transcript, max_shift=2.0)
        
        # Final validation to ensure quality
        snapped = validate_highlight_quality(snapped, importance_markers, transcript, video_duration)

        logger.info(f"Final highlight count: {len(snapped)}")
        total_final_duration = sum(h["end"] - h["start"] for h in snapped)
        logger.info(f"Total highlight duration: {total_final_duration:.1f} seconds")

        return snapped

    except Exception as e:
        logger.error(f"LLM error analyzing transcript: {e}")
        logger.info("Using fallback highlights.")
        return create_default_highlights(video_duration, target_num_highlights, segment_length, is_reel)

    except Exception as e:
        logger.error(f"LLM error analyzing transcript: {e}")
        logger.info("Using fallback highlights.")
        return create_default_highlights(video_duration, target_num_highlights, segment_length, is_reel)

def create_default_highlights(video_duration, num_segments=5, segment_duration=30, is_reel=False):
    """Create more intelligent default highlights if LLM analysis fails."""
    highlights = []
    
    # Better scaling formula based on video length
    if video_duration < 300:  # Under 5 minutes
        num_segments = max(3, min(5, int(video_duration / 60)))
        segment_duration = min(30, video_duration * 0.2)
    elif video_duration < 1200:  # 5-20 minutes
        num_segments = max(4, min(8, int(video_duration / 180)))
        segment_duration = min(45, video_duration * 0.1)
    else:  # Over 20 minutes
        num_segments = max(6, min(12, int(video_duration / 300)))
        segment_duration = min(60, video_duration * 0.05)
    
    # Always include beginning and end in default highlights
    # Beginning highlight
    highlights.append({
        "start": 0,
        "end": min(segment_duration, video_duration * 0.1),
        "description": "Introduction/Opening segment"
    })
    
    # Middle segments with slight randomization for better distribution
    middle_duration = video_duration - 2 * segment_duration
    if middle_duration > 0 and num_segments > 2:
        for i in range(num_segments - 2):
            # More natural positioning with golden ratio distribution
            pos = (i + 1) / (num_segments - 1) * middle_duration + segment_duration * 0.8
            # Add slight randomization for more natural feeling
            jitter = random.uniform(-0.05, 0.05) * middle_duration
            pos = max(segment_duration, min(video_duration - 2 * segment_duration, pos + jitter))
            highlights.append({
                "start": pos,
                "end": pos + segment_duration,
                "description": f"Key segment {i+1}"
            })
    
    # Ending highlight
    if video_duration > segment_duration * 1.5:
        highlights.append({
            "start": max(0, video_duration - segment_duration),
            "end": video_duration,
            "description": "Conclusion/Final segment"
        })
    
    # Sort by start time and ensure no overlaps
    highlights.sort(key=lambda x: x["start"])
    for i in range(1, len(highlights)):
        if highlights[i]["start"] < highlights[i-1]["end"]:
            highlights[i]["start"] = highlights[i-1]["end"]
            if highlights[i]["start"] >= highlights[i]["end"]:
                highlights[i]["end"] = min(highlights[i]["start"] + segment_duration, video_duration)
    
    return highlights

def extract_highlights(video_path, highlights):
    """Extract each highlight using MoviePy."""
    temp_dir = os.path.join(OUTPUT_DIR, "temp_clips")
    os.makedirs(temp_dir, exist_ok=True)

    def extract_single(idx, hl):
        s, e = hl["start"], hl["end"]
        out_file = os.path.join(temp_dir, f"highlight_{idx}.mp4")
        try:
            logger.info(f"Extracting clip {idx+1}/{len(highlights)}: {s:.1f}s–{e:.1f}s")
            # Use the correct import and method for MoviePy 2.0+
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            start_buffer = max(0, s - 0.2)
            # Use subclipped instead of subclip in MoviePy 2.0+
            clip = video.subclipped(start_buffer, e)
            clip.write_videofile(
                out_file,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast",
            )
            clip.close()
            video.close()

            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                return out_file, hl
            else:
                logger.error(f"Failed to create clip at {out_file} (empty).")
                return None, None
        except Exception as exc:
            logger.error(f"Error extracting clip {idx}: {exc}")
            return None, None

    try:
        successful_clips = []
        highlight_info = []

        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = {executor.submit(extract_single, i, hl): i for i, hl in enumerate(highlights)}
            for future in futures:
                clip_path, info = future.result()
                if clip_path:
                    successful_clips.append(clip_path)
                    highlight_info.append(info)

        # Check if we have any successful clips - FIXED INDENTATION
        if not successful_clips:
            fallback_clip = os.path.join(temp_dir, "highlight_default.mp4")
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            fallback_duration = min(30, video.duration)
            # Use subclipped instead of subclip
            clip = video.subclipped(0, fallback_duration)
            clip.write_videofile(
                fallback_clip,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast",
            )
            clip.close()
            video.close()

            return [fallback_clip], [{
                "start": 0,
                "end": fallback_duration,
                "description": "Default highlight"
            }]

        # Return successful clips - FIXED INDENTATION
        return successful_clips, highlight_info

    except Exception as e:
        logger.error(f"Error extracting highlights: {e}")
        return [], []

def merge_clips(clip_paths, highlight_info, is_reel=False):
    """Enhanced merging with better transitions based on content type."""
    logger.info(f"Merging {len(clip_paths)} clips with enhanced transitions...")
    
    if not clip_paths:
        raise ValueError("No clips to merge.")

    final_clips = []
    for cp in clip_paths:
        try:
            # Ensure the clips are correctly loaded as VideoFileClip
            clip = VideoFileClip(cp)
            final_clips.append(clip)
            logger.info(f"Loaded clip: {cp} (duration: {clip.duration:.2f}s)")
        except Exception as e:
            logger.error(f"Error loading clip {cp}: {e}")

    if not final_clips:
        raise ValueError("Failed to load any highlight clips.")
    
    logger.info(f"Successfully loaded {len(final_clips)} clips for merging")

    try:
        # Simple concatenation without transitions for standard highlights
        if not is_reel or len(final_clips) <= 1:
            logger.info(f"Concatenating {len(final_clips)} highlight clips...")
            merged_clip = concatenate_videoclips(final_clips, method="compose")
            output_path = os.path.join(OUTPUT_DIR, "final_highlights.mp4" if not is_reel else "reel.mp4")
        else:
            # For reels with transitions, try direct import of CrossFadeIn
            logger.info("Creating reel with transitions...")
            try:
                # Try importing directly first
                from moviepy.video.fx import crossfadein
                
                clips_w_transitions = [final_clips[0]]
                crossfade_dur = 0.5
                
                for c in final_clips[1:]:
                    # Apply crossfadein directly
                    transitioned_clip = crossfadein(c, crossfade_dur)
                    clips_w_transitions.append(transitioned_clip)
                
                merged_clip = concatenate_videoclips(clips_w_transitions, method="compose", padding=-crossfade_dur)
            except ImportError:
                # Fallback to simple concatenation if CrossFadeIn is not available
                logger.warning("CrossFadeIn not available, using simple concatenation")
                merged_clip = concatenate_videoclips(final_clips, method="compose")
            
            output_path = os.path.join(OUTPUT_DIR, "reel.mp4")

        # Write the merged video file
        logger.info(f"Writing merged video to {output_path}...")
        merged_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",  # Better quality/size balance than ultrafast
            bitrate="4000k"   # Higher quality output
        )
        
        # Ensure all clips are closed to release file handles
        merged_clip.close()
        for clip in final_clips:
            try:
                clip.close()
            except Exception as e:
                logger.error(f"Error closing clip: {e}")

        logger.info(f"Final video saved at: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error during video merging: {e}")
        
        # Try to close all clips even if there was an error
        for clip in final_clips:
            try:
                clip.close()
            except:
                pass
                
        # If merging failed but we have at least one clip, return the first one as fallback
        if len(final_clips) >= 1:
            logger.info("Merging failed. Returning first clip as fallback.")
            single_output = os.path.join(OUTPUT_DIR, "single_highlight.mp4")
            try:
                final_clips[0].write_videofile(
                    single_output,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    preset="medium",
                )
                final_clips[0].close()
                return single_output
            except Exception as e:
                logger.error(f"Error saving fallback clip: {e}")
                
        raise


def get_video_duration(video_path):
    """Get video duration using MoviePy."""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        logger.error(f"Failed to get video duration: {e}")
        return 300.0


async def generate_highlights(video_path, transcript_segments, video_info, target_duration=None, is_reel=False):
    """Main function to generate highlights from an already downloaded video."""
    try:
        print(f"Generating {'reel' if is_reel else 'video highlights'}. This may take a few minutes...")

        # Check if transcript_segments is None or empty
        if transcript_segments is None or len(transcript_segments) == 0:
            logger.warning("No transcript segments provided, creating default highlights.")
            vid_duration = get_video_duration(video_path)
            highlights = create_default_highlights(vid_duration)
            
            # Extract and merge clips with default highlights
            logger.info("Extracting highlight clips with defaults...")
            clip_paths, highlight_info = extract_highlights(video_path, highlights)
            
            logger.info("Merging clips into final video...")
            final_output = merge_clips(clip_paths, highlight_info, is_reel)
            
            return final_output, highlights

        # Convert transcript segments to the format needed
        transcript = []
        for start, end, text in transcript_segments:
            transcript.append({"text": text, "start": start, "end": end})

        # Get video duration
        vid_duration = get_video_duration(video_path)

        # Analyze for highlights
        logger.info("Analyzing transcript for highlights...")
        highlights = await analyze_transcript_for_highlights(
            transcript,
            video_info,
            vid_duration,
            target_total_duration=target_duration,
            is_reel=is_reel
        )

        # Extract and merge clips
        logger.info("Extracting highlight clips...")
        clip_paths, highlight_info = extract_highlights(video_path, highlights)

        logger.info("Merging clips into final video...")
        final_output = merge_clips(clip_paths, highlight_info, is_reel)

        # Clean up temp files
        temp_dir = os.path.join(OUTPUT_DIR, "temp_clips")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        return final_output, highlights

    except Exception as e:
        logger.error(f"Error generating highlights: {e}")
        return None, []
    

async def generate_custom_highlights(video_path, transcript_segments, video_info, user_input, target_duration=None):
    """Generate highlights based on custom user instructions and/or duration requirements."""
    start_time = time.time()
    
    # Convert transcript segments to the format needed
    transcript = []
    for start, end, text in transcript_segments:
        transcript.append({"text": text, "start": start, "end": end})
    
    # Get video duration
    vid_duration = get_video_duration(video_path)
    
    # Check if the user has specified key timestamps to include
    timestamp_pattern = r'(\d{1,2}[:]\d{1,2}[:]\d{0,2}|\d{1,2}[:]\d{1,2})'
    timestamp_mentions = re.findall(timestamp_pattern, user_input)
    
    # Convert mentioned timestamps to seconds
    timestamp_seconds = []
    for ts in timestamp_mentions:
        parts = ts.split(':')
        if len(parts) == 2:  # MM:SS format
            timestamp_seconds.append(int(parts[0]) * 60 + int(parts[1]))
        elif len(parts) == 3:  # HH:MM:SS format
            timestamp_seconds.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    
    # Determine key requests from the user input
    keep_first = any(term in user_input.lower() for term in ["beginning", "start", "intro", "first moment", "first part"])
    keep_last = any(term in user_input.lower() for term in ["ending", "end", "conclusion", "last moment", "last part"])
    
    # Calculate target duration if not provided
    if target_duration is None:
        # Default to 10-15% of video length
        target_duration = max(60, min(300, vid_duration * 0.15))
    
    # Build custom prompt for LLM
    custom_prompt = f"""
    You are an expert video editor. Create a highlight reel based on these specific requirements:
    
    VIDEO DETAILS:
    - Title: "{video_info.get('title', 'Unknown')}"
    - Duration: {format_time_duration(vid_duration)} ({vid_duration:.1f} s)
    - Description (snippet): "{(video_info.get('description') or '')[:300]}..."
    
    USER'S SPECIFIC INSTRUCTIONS:
    "{user_input}"
    
    HIGHLIGHT CRITERIA:
    1. Total highlight duration MUST be as close as possible to {target_duration:.1f} seconds (+/- 5 seconds).
    2. Create natural-feeling cuts between segments.
    3. Each highlight should be between 10-45 seconds long.
    """
    
    # Add custom requirements based on user's request
    if timestamp_seconds:
        timestamp_list = ", ".join([format_time_duration(ts) for ts in timestamp_seconds])
        custom_prompt += f"""
    4. IMPORTANT: Include segments that contain these specific timestamps: {timestamp_list}
        """
    
    if keep_first:
        custom_prompt += """
    5. MUST include the beginning/introduction of the video.
        """
    
    if keep_last:
        custom_prompt += """
    6. MUST include the ending/conclusion of the video.
        """
    
    # Add sample transcript and finalize prompt
    sample_segments = []
    if len(transcript) > 60:
        sample_segments.extend(transcript[:20])
        mid_idx = len(transcript)//2
        sample_segments.extend(transcript[mid_idx-10 : mid_idx+10])
        sample_segments.extend(transcript[-20:])
    else:
        sample_segments = transcript
    
    custom_prompt += f"""
    TRANSCRIPT SAMPLE (not entire transcript):
    {json.dumps(sample_segments, indent=2)}
    
    (N.B. The full transcript has {len(transcript)} segments.)
    
    INSTRUCTIONS:
    - Analyze the transcript and user's requirements.
    - Select segments that best match the user's specific request.
    - Ensure the total duration is very close to {target_duration:.1f} seconds.
    - Return ONLY a strict JSON array of highlight objects, no extra text.
    
    FINAL RESPONSE EXAMPLE:
    [
      {{
        "start": 10.0,
        "end": 25.0,
        "description": "Introduction explaining the core concept"
      }},
      ...
    ]
    """
    
    try:
        logger.info("Sending custom highlight request to LLM...")
        response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "system", "content": custom_prompt}])
        logger.info("Response received from LLM model.")
        raw_content = response["message"]["content"]
        
        # Process the response to extract valid JSON
        highlights = None
        
        # Try several JSON extraction methods (same as in the original code)
        try:
            highlights = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed, trying regex extraction...")
            
            # Try regex extraction
            json_match = re.search(r'\[\s*{.*}\s*\]', raw_content.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    highlights_str = json_match.group(0)
                    highlights = json.loads(highlights_str)
                except json.JSONDecodeError:
                    logger.warning("Regex JSON extraction failed")
            
            # If still no valid JSON, try to fix common issues
            if not highlights:
                logger.info("Trying to fix and extract JSON...")
                # Remove markdown code blocks
                cleaned = re.sub(r'```json\s*|\s*```', '', raw_content)
                # Try to find array bounds
                start_idx = cleaned.find('[')
                end_idx = cleaned.rfind(']')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    try:
                        json_str = cleaned[start_idx:end_idx+1]
                        highlights = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Fixed JSON extraction failed")
        
        # Validate the highlights
        if not highlights or not isinstance(highlights, list) or len(highlights) == 0:
            logger.warning("Empty or invalid highlights from LLM. Using fallback.")
            return create_default_highlights(vid_duration, 
                                            num_segments=max(3, int(target_duration / 30)),
                                            segment_duration=min(30, target_duration / 3),
                                            is_reel=False)
        
        # Process and clean the highlights
        cleaned = []
        total_hl_duration = 0
        
        # Ensure required timestamps are included if specified
        if timestamp_seconds and highlights:
            # Check if any of the required timestamps are missing
            covered_timestamps = set()
            for h in highlights:
                start, end = float(h["start"]), float(h["end"])
                for ts in timestamp_seconds:
                    if start <= ts <= end:
                        covered_timestamps.add(ts)
            
            # For any missing timestamps, add a segment centered on that timestamp
            for ts in set(timestamp_seconds) - covered_timestamps:
                segment_start = max(0, ts - 10)  # 10 seconds before
                segment_end = min(vid_duration, ts + 10)  # 10 seconds after
                
                # Don't add if too short
                if segment_end - segment_start >= 5:
                    highlights.append({
                        "start": segment_start,
                        "end": segment_end,
                        "description": f"Requested timestamp at {format_timestamp(ts)}"
                    })
        
        # Clean up & validate each highlight
        for h in highlights:
            if not all(k in h for k in ("start", "end", "description")):
                continue
            s, e = float(h["start"]), float(h["end"])
            s = max(0, s)
            e = min(vid_duration, e)
            if e <= s or (e - s) < 2:
                continue
            if (e - s) > 60:  # Cap segment length at 60 seconds
                e = s + 60
            cleaned.append({
                "start": s,
                "end": e,
                "description": h["description"].strip()
            })
            total_hl_duration += (e - s)
        
        # Sort by start time
        cleaned.sort(key=lambda x: x["start"])
        
        # Merge overlapping segments
        merged = []
        for c in cleaned:
            if not merged or c["start"] > merged[-1]["end"]:
                merged.append(c)
            else:
                merged[-1]["end"] = max(merged[-1]["end"], c["end"])
                merged[-1]["description"] += " + " + c["description"]
        
        # Check if duration constraints are met
        total_duration = sum(h["end"] - h["start"] for h in merged)
        
        # Adjust if the total duration is off by more than 15%
        if abs(total_duration - target_duration) > (target_duration * 0.15):
            logger.info(f"Duration mismatch: {total_duration:.1f}s vs target {target_duration:.1f}s. Adjusting...")
            
            if total_duration > target_duration:
                # Too long - remove least important segments or shorten segments
                excess = total_duration - target_duration
                
                # Try to shorten segments proportionally first
                if len(merged) > 0:
                    # Calculate how much to trim from each segment
                    trim_per_segment = excess / len(merged)
                    for h in merged:
                        segment_duration = h["end"] - h["start"]
                        if segment_duration > 10:  # Only trim segments longer than 10s
                            trim_amount = min(segment_duration - 10, trim_per_segment)
                            h["end"] -= trim_amount
                    
                # If still too long, remove segments (keeping required ones)
                total_duration = sum(h["end"] - h["start"] for h in merged)
                if total_duration > target_duration * 1.05:
                    # Identify segments to keep
                    must_keep = []
                    can_remove = []
                    
                    for i, h in enumerate(merged):
                        # Keep first segment if requested
                        if i == 0 and keep_first:
                            must_keep.append(i)
                            continue
                        
                        # Keep last segment if requested
                        if i == len(merged)-1 and keep_last:
                            must_keep.append(i)
                            continue
                            
                        # Keep segments with requested timestamps
                        keep_this = False
                        for ts in timestamp_seconds:
                            if h["start"] <= ts <= h["end"]:
                                keep_this = True
                                break
                        
                        if keep_this:
                            must_keep.append(i)
                        else:
                            can_remove.append((i, h["end"] - h["start"]))
                    
                    # Sort removable segments by duration (shortest first)
                    can_remove.sort(key=lambda x: x[1])
                    
                    # Remove segments until we're under target
                    new_merged = [merged[i] for i in range(len(merged))]
                    for idx, _ in can_remove:
                        if sum(h["end"] - h["start"] for h in new_merged) <= target_duration:
                            break
                        new_merged[idx] = None
                    
                    merged = [h for h in new_merged if h is not None]
            
            elif total_duration < target_duration * 0.85:
                # Too short - extend segments or add more
                shortfall = target_duration - total_duration
                
                # Try to extend existing segments first
                extend_per_segment = shortfall / len(merged)
                for h in merged:
                    # Extend each segment by equal amount, up to limit
                    max_extension = min(15, extend_per_segment)  # Don't extend by more than 15s
                    h["end"] = min(vid_duration, h["end"] + max_extension)
        
        # Final sorting and merging pass
        merged.sort(key=lambda x: x["start"])
        final_segments = []
        for c in merged:
            if not final_segments or c["start"] > final_segments[-1]["end"]:
                final_segments.append(c)
            else:
                final_segments[-1]["end"] = max(final_segments[-1]["end"], c["end"])
                final_segments[-1]["description"] += " + " + c["description"]
        
        # Snap to transcript boundaries for natural cut points
        snapped = snap_highlights_to_transcript_boundaries(final_segments, transcript, max_shift=2.0)
        
        total_final_duration = sum(h["end"] - h["start"] for h in snapped)
        logger.info(f"Custom highlights generated: {len(snapped)} segments")
        logger.info(f"Total highlight duration: {total_final_duration:.1f} seconds (target: {target_duration:.1f}s)")
        
        return snapped
        
    except Exception as e:
        logger.error(f"Error generating custom highlights: {e}")
        return create_default_highlights(vid_duration, 
                                        num_segments=max(3, int(target_duration / 30)),
                                        segment_duration=min(30, target_duration / 3),
                                        is_reel=False)



##################################   Interactive    Q/A  ##################################################################################

def extract_qa_clips(video_path, highlights, transcript_segments):
    """Extract precise clips for Q&A functionality using MoviePy without fade effects."""
    temp_dir = os.path.join(OUTPUT_DIR, "temp_qa_clips")
    os.makedirs(temp_dir, exist_ok=True)

    def extract_single(idx, hl):
        s, e = hl["start"], hl["end"]
        out_file = os.path.join(temp_dir, f"qa_clip_{idx}.mp4")
        try:
            logger.info(f"Extracting Q&A clip {idx+1}/{len(highlights)}: {s:.1f}s–{e:.1f}s")
            # Use the correct import and method for MoviePy 2.0+
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            
            # Skip intro segments unless explicitly requested
            if s < 30 and not "introduction" in hl["description"].lower():
                logger.info(f"Skipping intro segment at {s:.1f}s")
                video.close()
                return None, None
            
            # Load a buffer for smoother cutting
            load_buffer = 1.0  # Increased buffer for smoother audio transitions
            start_load = max(0, s - load_buffer)
            end_load = min(video.duration, e + load_buffer)
            
            # First load a slightly larger segment to ensure smooth cutting
            working_clip = video.subclipped(start_load, end_load)
            
            # Then precisely cut at the exact timestamps (relative to the working clip's start)
            precise_start = s - start_load  # Relative position in the working clip
            precise_end = e - start_load    # Relative position in the working clip
            
            # Ensure we don't get negative values if s was 0
            precise_start = max(0, precise_start)
            
            # Create the final clip with precise boundaries
            final_clip = working_clip.subclipped(precise_start, precise_end)
            
            # Write to file
            final_clip.write_videofile(
                out_file,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast",
            )
            
            # Clean up
            final_clip.close()
            working_clip.close()
            video.close()

            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                return out_file, hl
            else:
                logger.error(f"Failed to create Q&A clip at {out_file} (empty).")
                return None, None
        except Exception as exc:
            logger.error(f"Error extracting Q&A clip {idx}: {exc}")
            return None, None

    try:
        successful_clips = []
        highlight_info = []

        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = {executor.submit(extract_single, i, hl): i for i, hl in enumerate(highlights)}
            for future in futures:
                clip_path, info = future.result()
                if clip_path:
                    successful_clips.append(clip_path)
                    highlight_info.append(info)

        # Check if we have any successful clips
        if not successful_clips:
            # If no successful clips, create a single meaningful clip
            logger.warning("No Q&A clips were successfully extracted. Creating a fallback clip.")
            fallback_clip = os.path.join(temp_dir, "qa_fallback.mp4")
            
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            
            # Find a meaningful segment from transcript
            fallback_segment = None
            for start, end, text in transcript_segments:
                # Skip intro
                if start >= 30 and end - start >= 10 and re.search(r'[.!?]$', text):
                    fallback_segment = (start, end, text)
                    break
            
            if fallback_segment:
                fallback_start, fallback_end, _ = fallback_segment
            else:
                # Default to middle of video if no good segment found
                video_middle = video.duration / 2
                fallback_start = max(30, video_middle - 15)
                fallback_end = fallback_start + 30
            
            # Use subclipped
            clip = video.subclipped(fallback_start, fallback_end)
            clip.write_videofile(
                fallback_clip,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast",
            )
            clip.close()
            video.close()

            return [fallback_clip], [{
                "start": fallback_start,
                "end": fallback_end,
                "description": "Complete sentence from video"
            }]

        # Return successful clips
        return successful_clips, highlight_info

    except Exception as e:
        logger.error(f"Error extracting Q&A clips: {e}")
        return [], []


def merge_qa_clips(clip_paths, highlight_info, is_reel=False):
    """Merge Q&A clips with precise boundaries and better naming - no transitions."""
    logger.info(f"Merging {len(clip_paths)} Q&A clips...")
    
    if not clip_paths:
        raise ValueError("No Q&A clips to merge.")

    # Generate a better output filename based on content
    output_filename = "qa_answer.mp4"
    if highlight_info and len(highlight_info) > 0:
        # Try to create a descriptive filename
        if "description" in highlight_info[0]:
            # Use the first description as the basis for the filename
            desc = highlight_info[0]["description"]
            if desc.startswith("Contains: "):
                desc = desc[10:]  # Remove "Contains: " prefix
            # Clean it up for a filename
            safe_desc = re.sub(r'[^\w\s-]', '', desc).strip().replace(' ', '_')[:40]
            output_filename = f"qa_{safe_desc}.mp4"
    
    # Designate output path
    output_dir = os.path.join(OUTPUT_DIR, "qa_clips")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Load clips
    final_clips = []
    for cp in clip_paths:
        try:
            # Ensure the clips are correctly loaded as VideoFileClip
            clip = VideoFileClip(cp)
            final_clips.append(clip)
            logger.info(f"Loaded Q&A clip: {cp} (duration: {clip.duration:.2f}s)")
        except Exception as e:
            logger.error(f"Error loading Q&A clip {cp}: {e}")

    if not final_clips:
        raise ValueError("Failed to load any Q&A clips.")
    
    logger.info(f"Successfully loaded {len(final_clips)} Q&A clips for merging")

    try:
        # Simple concatenation for Q&A clips - no transitions
        logger.info(f"Concatenating {len(final_clips)} Q&A clips (no transitions)...")
        merged_clip = concatenate_videoclips(final_clips, method="compose")

        # Write the merged video file
        logger.info(f"Writing merged Q&A video to {output_path}...")
        merged_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",  # Better quality/size balance than ultrafast
            bitrate="4000k"   # Higher quality output
        )
        
        # Ensure all clips are closed to release file handles
        merged_clip.close()
        for clip in final_clips:
            try:
                clip.close()
            except Exception as e:
                logger.error(f"Error closing Q&A clip: {e}")

        logger.info(f"Final Q&A video saved at: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error during Q&A video merging: {e}")
        
        # Try to close all clips even if there was an error
        for clip in final_clips:
            try:
                clip.close()
            except:
                pass
                
        # If merging failed but we have at least one clip, return the first one as fallback
        if len(final_clips) >= 1:
            logger.info("Q&A merging failed. Returning first clip as fallback.")
            single_output = os.path.join(output_dir, "qa_single_clip.mp4")
            try:
                final_clips[0].write_videofile(
                    single_output,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    preset="medium",
                )
                final_clips[0].close()
                return single_output
            except Exception as e:
                logger.error(f"Error saving Q&A fallback clip: {e}")
                
        raise