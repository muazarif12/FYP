import os
import asyncio
import json
import time
import re
from logger_config import logger
from utils import format_timestamp
from ffmpeg_check import verify_dependencies
from dubbing import translate_transcript_to_english  # Reuse translation function

async def create_subtitle_file(translated_segments, output_dir, subtitle_format="srt"):
    """
    Create a subtitle file from translated transcript segments.
    
    Args:
        translated_segments: List of tuples (start_time, end_time, translated_text)
        output_dir: Directory to save subtitle file
        subtitle_format: Format of subtitle file (srt, vtt)
        
    Returns:
        Path to the subtitle file
    """
    try:
        # Verify ffmpeg is available
        deps_ok, deps_message = verify_dependencies()
        if not deps_ok:
            logger.error(f"Dependency check failed: {deps_message}")
            return None
        
        # Create subtitles directory
        subtitles_dir = os.path.join(output_dir, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        
        # Generate file paths
        timestamp = int(time.time())
        srt_path = os.path.join(subtitles_dir, f"english_subtitles_{timestamp}.srt")
        vtt_path = os.path.join(subtitles_dir, f"english_subtitles_{timestamp}.vtt")
        
        # Sort segments by start time to ensure proper sequence
        sorted_segments = sorted(translated_segments, key=lambda x: x[0])
        
        # Generate SRT file (SubRip format)
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for i, (start, end, text) in enumerate(sorted_segments, 1):
                # Format timestamps as HH:MM:SS,mmm
                start_time_srt = format_srt_timestamp(start)
                end_time_srt = format_srt_timestamp(end)
                
                # Write subtitle entry
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
                srt_file.write(f"{text}\n\n")
        
        # Generate WebVTT file for web players
        with open(vtt_path, 'w', encoding='utf-8') as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for i, (start, end, text) in enumerate(sorted_segments, 1):
                # Format timestamps as HH:MM:SS.mmm
                start_time_vtt = format_vtt_timestamp(start)
                end_time_vtt = format_vtt_timestamp(end)
                
                # Write subtitle entry
                vtt_file.write(f"{i}\n")
                vtt_file.write(f"{start_time_vtt} --> {end_time_vtt}\n")
                vtt_file.write(f"{text}\n\n")
        
        # Return the path to the requested format
        if subtitle_format.lower() == "vtt":
            return vtt_path
        else:
            return srt_path
            
    except Exception as e:
        logger.error(f"Error creating subtitle file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

def format_vtt_timestamp(seconds):
    """Convert seconds to WebVTT timestamp format: HH:MM:SS.mmm"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"

# async def embed_subtitles_in_video(video_path, subtitle_path, output_dir):
#     """
#     Embed subtitles into a video file.
    
#     Args:
#         video_path: Path to the video file
#         subtitle_path: Path to the subtitle file
#         output_dir: Directory to save the output video
        
#     Returns:
#         Path to the subtitled video
#     """
#     try:
#         # Create output directory
#         subtitled_dir = os.path.join(output_dir, "subtitled")
#         os.makedirs(subtitled_dir, exist_ok=True)
        
#         # Generate output file path
#         video_filename = os.path.basename(video_path)
#         base_name, ext = os.path.splitext(video_filename)
#         output_path = os.path.join(subtitled_dir, f"{base_name}_subtitled{ext}")
        
#         # Determine subtitle format from file extension
#         _, subtitle_ext = os.path.splitext(subtitle_path)
#         subtitle_format = subtitle_ext.lower()[1:]  # Remove the dot
        
#         # Use ffmpeg to embed subtitles
#         ffmpeg_command = [
#             "ffmpeg", "-y",
#             "-i", video_path,
#             "-i", subtitle_path,
#             "-c:v", "copy",  # Copy video stream without re-encoding
#             "-c:a", "copy",  # Copy audio stream without re-encoding
#             "-c:s", "mov_text" if ext.lower() == '.mp4' else "srt",  # Choose subtitle codec based on container
#             "-metadata:s:s:0", f"language=eng",  # Set subtitle language metadata
#             "-disposition:s:0", "default",  # Mark subtitle as default
#             output_path
#         ]
        
#         # Execute ffmpeg command
#         logger.info("Embedding subtitles into video...")
#         process = await asyncio.create_subprocess_exec(
#             *ffmpeg_command,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         stdout, stderr = await process.communicate()
        
#         if process.returncode != 0:
#             logger.error(f"Error embedding subtitles: {stderr.decode()}")
            
#             # Try alternative approach (hardcode subtitles)
#             logger.info("Trying alternative method (hardcoding subtitles)...")
#             alt_command = [
#                 "ffmpeg", "-y",
#                 "-i", video_path,
#                 "-vf", f"subtitles={subtitle_path}",
#                 "-c:a", "copy",  # Copy audio without re-encoding
#                 output_path
#             ]
            
#             process = await asyncio.create_subprocess_exec(
#                 *alt_command,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )
            
#             stdout, stderr = await process.communicate()
            
#             if process.returncode != 0:
#                 logger.error(f"Alternative subtitling failed: {stderr.decode()}")
#                 return None
        
#         logger.info(f"Subtitled video created successfully: {output_path}")
#         return output_path
        
#     except Exception as e:
#         logger.error(f"Error embedding subtitles: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None


async def embed_subtitles_in_video(video_path, subtitle_path, output_dir):
    """
    Embed subtitles into a video file.
    
    Args:
        video_path: Path to the video file
        subtitle_path: Path to the subtitle file
        output_dir: Directory to save the output video
        
    Returns:
        Path to the subtitled video
    """
    try:
        # Create output directory
        subtitled_dir = os.path.join(output_dir, "subtitled")
        os.makedirs(subtitled_dir, exist_ok=True)
        
        # Generate output file path
        video_filename = os.path.basename(video_path)
        base_name, ext = os.path.splitext(video_filename)
        output_path = os.path.join(subtitled_dir, f"{base_name}_subtitled{ext}")
        
        # Skip the first method that just embeds subtitles as a separate stream
        # Use subtitle filter directly to burn-in/hardcode subtitles
        logger.info("Hardcoding subtitles into video...")
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles={subtitle_path}",
            "-c:a", "copy",  # Copy audio without re-encoding
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Subtitling failed: {stderr.decode()}")
            return None
        
        logger.info(f"Subtitled video created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error embedding subtitles: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def create_english_subtitles(video_path, transcript_segments, detected_language, output_dir):
    """
    Main function to create English subtitles for a non-English video.
    
    Args:
        video_path: Path to the video file
        transcript_segments: List of tuples (start_time, end_time, text)
        detected_language: The detected language code
        output_dir: Directory to save the output
        
    Returns:
        Tuple of (subtitled_video_path, subtitle_path, stats)
    """
    try:
        # Skip if already in English
        if detected_language == "en":
            logger.info("Video is already in English, skipping subtitle creation")
            return None, None, "Video is already in English, no subtitles needed."
        
        start_time = time.time()
        logger.info(f"Starting English subtitle creation for {detected_language} video...")
        
        # Step 1: Translate transcript to English (reuse the function from dubbing)
        logger.info("Step 1/3: Translating transcript to English...")
        translated_segments = await translate_transcript_to_english(transcript_segments, detected_language)
        
        if not translated_segments:
            return None, None, "Failed to translate transcript to English."
        
        # Step 2: Create subtitle files
        logger.info("Step 2/3: Creating subtitle files...")
        subtitle_path = await create_subtitle_file(translated_segments, output_dir)
        
        if not subtitle_path:
            return None, None, "Failed to create subtitle files."
        
        # Step 3: Embed subtitles in video
        logger.info("Step 3/3: Embedding subtitles in video...")
        subtitled_video = await embed_subtitles_in_video(video_path, subtitle_path, output_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not subtitled_video:
            # Return subtitle file even if embedding failed
            logger.info("Could not embed subtitles, but subtitle file was created successfully")
            stats = {
                "original_language": detected_language,
                "segments_translated": len(translated_segments),
                "subtitle_file": subtitle_path,
                "processing_time": f"{processing_time:.2f} seconds"
            }
            return None, subtitle_path, stats
        
        # Return success stats
        stats = {
            "original_language": detected_language,
            "segments_translated": len(translated_segments),
            "subtitle_file": subtitle_path,
            "subtitled_video": subtitled_video,
            "processing_time": f"{processing_time:.2f} seconds"
        }
        
        logger.info(f"English subtitling completed in {processing_time:.2f} seconds")
        return subtitled_video, subtitle_path, stats
        
    except Exception as e:
        logger.error(f"Error in English subtitling process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, f"Error creating English subtitles: {str(e)}"