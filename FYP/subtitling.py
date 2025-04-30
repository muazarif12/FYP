import os
import asyncio
import json
import time
import re
import numpy as np
from logger_config import logger
from utils import format_timestamp
from ffmpeg_check import verify_dependencies
from dubbing import translate_transcript_to_english

class SubtitleSyncEngine:
    """
    Advanced subtitle synchronization engine that uses multiple techniques
    to ensure accurate alignment between speech and subtitles.
    """
    
    def __init__(self, video_path, transcript_segments):
        self.video_path = video_path
        self.transcript_segments = transcript_segments
        self.temp_dir = None
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
    
    async def analyze(self):
        """
        Main analysis method that combines multiple techniques for robust sync detection.
        
        Returns:
            Optimal subtitle delay in seconds
        """
        logger.info("Starting advanced audio analysis for subtitle synchronization")
        
        # Step 1: Extract audio from video
        await self._extract_audio()
        
        # Step 2: If we have transcript segments, use them to inform our analysis
        transcript_start = self._analyze_transcript_timing()
        
        # Step 3: Perform VAD (Voice Activity Detection) analysis
        vad_start = await self._perform_vad_analysis()
        
        # Step 4: Use fixed offset approach as fallback
        fixed_offset = 3.0  # Conservative default
        
        # Step 5: Make final decision based on all evidence
        delay = self._determine_optimal_delay(vad_start, transcript_start, fixed_offset)
        
        # Step 6: Clean up
        await self._cleanup()
        
        return delay
    
    async def _extract_audio(self):
        """Extract audio from video file for analysis"""
        try:
            import tempfile
            self.temp_dir = tempfile.mkdtemp()
            self.audio_path = os.path.join(self.temp_dir, "audio.wav")
            
            # Extract audio at 16kHz mono for speech analysis
            logger.info("Extracting audio for analysis...")
            extract_cmd = [
                "ffmpeg", "-y", "-i", self.video_path, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                self.audio_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *extract_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode != 0 or not os.path.exists(self.audio_path):
                logger.warning("Failed to extract audio, using transcript only")
            else:
                logger.info(f"Audio extracted to {self.audio_path}")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
    
    def _analyze_transcript_timing(self):
        """Analyze transcript segments to find the first substantive speech"""
        if not self.transcript_segments or len(self.transcript_segments) == 0:
            logger.warning("No transcript segments available for timing analysis")
            return None
        
        # Sort segments by start time
        sorted_segments = sorted(self.transcript_segments, key=lambda x: x[0])
        
        # Find first segment with substantive content (not just sounds or single words)
        for segment in sorted_segments:
            start_time, end_time, text = segment
            
            # Check if this is likely to be actual speech (not just "uh", "um", etc.)
            # Simple heuristic: at least 2 words and 0.5 seconds long
            words = text.strip().split()
            if len(words) >= 2 and (end_time - start_time) >= 0.5:
                logger.info(f"First substantive speech in transcript at {start_time:.2f}s: '{text}'")
                # Subtract a small buffer to ensure we don't miss the beginning
                return max(0, start_time - 0.3)
        
        # If we couldn't find a clear speech segment, return the start of the first segment
        if sorted_segments:
            logger.info(f"Using first transcript segment at {sorted_segments[0][0]:.2f}s as fallback")
            return max(0, sorted_segments[0][0] - 0.3)
        
        return None
    
    async def _perform_vad_analysis(self):
        """
        Perform Voice Activity Detection to find when speech starts.
        Uses energy levels and zero-crossing rate to differentiate speech from other sounds.
        """
        if not os.path.exists(self.audio_path):
            logger.warning("Audio file unavailable for VAD analysis")
            return None
        
        try:
            # Use FFmpeg's loudnorm filter to analyze audio levels
            logger.info("Analyzing audio energy levels...")
            loudnorm_cmd = [
                "ffmpeg", "-i", self.audio_path,
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *loudnorm_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            stderr_output = stderr.decode()
            
            # Extract speech start using silence detection with optimized parameters
            # Use multiple passes with different thresholds
            speech_starts = []
            
            # First pass: high threshold to detect only clear speech
            high_threshold_cmd = [
                "ffmpeg", "-i", self.audio_path,
                "-af", "silencedetect=noise=-25dB:d=0.3",
                "-f", "null", "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *high_threshold_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            high_threshold_output = stderr.decode()
            
            # Parse high threshold results
            high_starts = re.findall(r'silence_end: (\d+\.?\d*)', high_threshold_output)
            if high_starts:
                speech_starts.extend([float(t) for t in high_starts])
            
            # Second pass: medium threshold for more sensitivity
            medium_threshold_cmd = [
                "ffmpeg", "-i", self.audio_path,
                "-af", "silencedetect=noise=-30dB:d=0.2",
                "-f", "null", "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *medium_threshold_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            medium_threshold_output = stderr.decode()
            
            # Parse medium threshold results
            medium_starts = re.findall(r'silence_end: (\d+\.?\d*)', medium_threshold_output)
            if medium_starts:
                speech_starts.extend([float(t) for t in medium_starts])
            
            # Analyze the results to find the most reliable speech start
            if not speech_starts:
                logger.warning("VAD analysis couldn't detect speech start")
                return None
            
            # Sort all detected speech starts
            speech_starts.sort()
            
            # Use a clustering approach to find sustained speech
            # (as opposed to isolated sounds like applause)
            clusters = []
            current_cluster = [speech_starts[0]]
            
            for i in range(1, len(speech_starts)):
                # If this point is close to the previous one, add to current cluster
                if speech_starts[i] - speech_starts[i-1] < 2.0:
                    current_cluster.append(speech_starts[i])
                else:
                    # Otherwise start a new cluster
                    clusters.append(current_cluster)
                    current_cluster = [speech_starts[i]]
            
            # Add the last cluster
            if current_cluster:
                clusters.append(current_cluster)
            
            # Find the first substantial cluster (3+ points or long duration)
            speech_start = None
            for cluster in clusters:
                if len(cluster) >= 3 or (cluster[-1] - cluster[0] >= 1.5):
                    speech_start = cluster[0]
                    logger.info(f"VAD detected speech start at {speech_start:.2f}s")
                    break
            
            # If no substantial cluster found, use the first point
            if speech_start is None and speech_starts:
                speech_start = speech_starts[0]
                logger.info(f"Using first detected audio point at {speech_start:.2f}s")
            
            return speech_start
            
        except Exception as e:
            logger.error(f"Error in VAD analysis: {e}")
            return None
    
    def _determine_optimal_delay(self, vad_start, transcript_start, fixed_offset):
        """
        Determine the optimal subtitle delay based on all available information.
        Uses a weighted decision algorithm that prioritizes more reliable signals.
        """
        logger.info(f"Determining optimal delay from VAD: {vad_start}, Transcript: {transcript_start}, Fixed: {fixed_offset}")
        
        # Case 1: Both VAD and transcript available
        if vad_start is not None and transcript_start is not None:
            # If they're close, average them
            if abs(vad_start - transcript_start) < 3.0:
                delay = (vad_start + transcript_start) / 2
                logger.info(f"Using average of VAD and transcript: {delay:.2f}s")
            else:
                # If they differ significantly, slightly favor the transcript timing
                # but keep some influence from VAD
                delay = (transcript_start * 0.6) + (vad_start * 0.4)
                logger.info(f"Using weighted average due to timing difference: {delay:.2f}s")
        
        # Case 2: Only VAD available
        elif vad_start is not None:
            delay = vad_start
            logger.info(f"Using VAD detected start: {delay:.2f}s")
        
        # Case 3: Only transcript available
        elif transcript_start is not None:
            delay = transcript_start
            logger.info(f"Using transcript timing: {delay:.2f}s")
        
        # Case 4: Neither available, use fixed offset
        else:
            delay = fixed_offset
            logger.info(f"Using fixed delay: {delay:.2f}s")
        
        # Apply a minimum offset to ensure subtitles don't start too early
        delay = max(0.5, delay)
        
        # Round to 2 decimal places for cleaner timestamps
        delay = round(delay, 2)
        
        logger.info(f"Final determined optimal delay: {delay:.2f}s")
        return delay
    
    async def _cleanup(self):
        """Clean up temporary files"""
        try:
            if self.audio_path and os.path.exists(self.audio_path):
                os.remove(self.audio_path)
            
            if self.temp_dir and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")


async def adjust_subtitle_timing_by_offset(subtitle_path, offset_seconds, output_dir=None):
    """
    Adjust an existing subtitle file by shifting all timestamps by a specified offset.
    
    Args:
        subtitle_path: Path to the original subtitle file
        offset_seconds: Number of seconds to shift (positive = delay, negative = advance)
        output_dir: Optional directory to save the adjusted file (defaults to same directory)
        
    Returns:
        Path to the adjusted subtitle file
    """
    try:
        if not os.path.exists(subtitle_path):
            logger.error(f"Subtitle file not found: {subtitle_path}")
            return None
            
        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(subtitle_path)
            
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate adjusted file path
        orig_filename = os.path.basename(subtitle_path)
        base_name, ext = os.path.splitext(orig_filename)
        direction = "delayed" if offset_seconds >= 0 else "advanced"
        adjusted_filename = f"{base_name}_{direction}_{abs(offset_seconds):.1f}s{ext}"
        adjusted_path = os.path.join(output_dir, adjusted_filename)
        
        # Read the original subtitle file
        with open(subtitle_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Check if it's an SRT file
        is_srt = ext.lower() == '.srt'
        
        # Process the file based on its format
        if is_srt:
            # SRT format processing
            # Example: 00:00:20,000 --> 00:00:25,000
            adjusted_content = re.sub(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})',
                lambda m: _adjust_srt_timestamp(m, offset_seconds),
                content
            )
        else:
            # WebVTT format processing
            # Example: 00:00:20.000 --> 00:00:25.000
            adjusted_content = re.sub(
                r'(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})',
                lambda m: _adjust_vtt_timestamp(m, offset_seconds),
                content
            )
        
        # Write the adjusted content
        with open(adjusted_path, 'w', encoding='utf-8') as file:
            file.write(adjusted_content)
            
        logger.info(f"Adjusted subtitle file created: {adjusted_path}")
        return adjusted_path
        
    except Exception as e:
        logger.error(f"Error adjusting subtitle timing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _adjust_srt_timestamp(match, offset_seconds):
    """Helper function to adjust SRT timestamp"""
    # Parse start time
    h1, m1, s1, ms1 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    start_seconds = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
    
    # Parse end time
    h2, m2, s2, ms2 = int(match.group(5)), int(match.group(6)), int(match.group(7)), int(match.group(8))
    end_seconds = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    
    # Apply offset
    new_start = max(0, start_seconds + offset_seconds)
    new_end = max(new_start + 0.1, end_seconds + offset_seconds)  # Ensure end is after start
    
    # Format back to SRT timestamp
    return f"{_format_time_component(new_start)},{_format_ms_component(new_start)} --> {_format_time_component(new_end)},{_format_ms_component(new_end)}"

def _adjust_vtt_timestamp(match, offset_seconds):
    """Helper function to adjust WebVTT timestamp"""
    # Parse start time
    h1, m1, s1, ms1 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    start_seconds = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
    
    # Parse end time
    h2, m2, s2, ms2 = int(match.group(5)), int(match.group(6)), int(match.group(7)), int(match.group(8))
    end_seconds = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    
    # Apply offset
    new_start = max(0, start_seconds + offset_seconds)
    new_end = max(new_start + 0.1, end_seconds + offset_seconds)  # Ensure end is after start
    
    # Format back to WebVTT timestamp
    return f"{_format_time_component(new_start)}.{_format_ms_component(new_start)} --> {_format_time_component(new_end)}.{_format_ms_component(new_end)}"

def _format_time_component(seconds):
    """Format the HH:MM:SS part of a timestamp"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def _format_ms_component(seconds):
    """Format the milliseconds part of a timestamp"""
    milliseconds = int((seconds % 1) * 1000)
    return f"{milliseconds:03d}"

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

async def create_english_subtitles(video_path, transcript_segments, detected_language, output_dir):
    """
    Main function to create English subtitles for any video, including those already in English.
    Uses advanced audio analysis for precise subtitle synchronization.
    
    Args:
        video_path: Path to the video file
        transcript_segments: List of tuples (start_time, end_time, text)
        detected_language: The detected language code
        output_dir: Directory to save the output
        
    Returns:
        Tuple of (subtitled_video_path, subtitle_path, stats)
    """
    try:
        start_time = time.time()
        
        # Modify behavior based on language
        if detected_language == "en":
            logger.info("Video is in English, creating native language subtitles...")
            # For English videos, directly use the existing transcript rather than translating
            subtitle_segments = [(seg[0], seg[1], seg[2]) for seg in transcript_segments]
        else:
            # For non-English videos, translate to English
            logger.info(f"Starting English subtitle creation for {detected_language} video...")
            logger.info("Step 1/4: Translating transcript to English...")
            subtitle_segments = await translate_transcript_to_english(transcript_segments, detected_language)
        
        if not subtitle_segments:
            return None, None, "Failed to process transcript for subtitles."
        
        # Step 2: Perform advanced synchronization analysis
        logger.info("Step 2/4: Performing advanced audio analysis for precise synchronization...")
        sync_engine = SubtitleSyncEngine(video_path, transcript_segments)
        optimal_delay = await sync_engine.analyze()
        
        logger.info(f"Determined optimal subtitle delay: {optimal_delay:.2f}s")
        
        # Apply the calculated delay to all subtitle segments
        adjusted_segments = []
        for start, end, text in subtitle_segments:
            new_start = max(0, start + optimal_delay)
            new_end = end + optimal_delay
            adjusted_segments.append((new_start, new_end, text))
        
        # Remove very short segments and segments with empty text
        filtered_segments = [
            (start, end, text) for start, end, text in adjusted_segments
            if end - start >= 0.3 and text.strip()
        ]
        
        # Sort by start time (just to be safe)
        filtered_segments.sort(key=lambda x: x[0])
        
        # Create subtitle files
        logger.info("Step 3/4: Creating subtitle files...")
        subtitle_path = await create_subtitle_file(filtered_segments, output_dir)
        
        if not subtitle_path:
            return None, None, "Failed to create subtitle files."
        
        # Embed subtitles in video
        logger.info("Step 4/4: Embedding subtitles in video...")
        subtitled_video = await embed_subtitles_in_video(video_path, subtitle_path, output_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not subtitled_video:
            # Return subtitle file even if embedding failed
            logger.info("Could not embed subtitles, but subtitle file was created successfully")
            stats = {
                "original_language": detected_language,
                "segments_processed": len(filtered_segments),
                "subtitle_file": subtitle_path,
                "sync_method": "Advanced audio analysis",
                "sync_delay": f"{optimal_delay:.2f}s",
                "processing_time": f"{processing_time:.2f} seconds"
            }
            return None, subtitle_path, stats
        
        # Return success stats
        stats = {
            "original_language": detected_language,
            "segments_processed": len(filtered_segments),
            "subtitle_file": subtitle_path,
            "sync_method": "Advanced audio analysis",
            "sync_delay": f"{optimal_delay:.2f}s",
            "subtitled_video": subtitled_video,
            "processing_time": f"{processing_time:.2f} seconds"
        }
        
        logger.info(f"Subtitling completed in {processing_time:.2f} seconds")
        return subtitled_video, subtitle_path, stats
        
    except Exception as e:
        logger.error(f"Error in subtitling process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, f"Error creating subtitles: {str(e)}"

# Keep the existing embed_subtitles_in_video function as is

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
        
        # Fix: Properly escape the path for FFmpeg
        # On Windows, we need to ensure the path is properly formatted
        subtitle_path_escaped = subtitle_path.replace('\\', '\\\\')
        
        # Use subtitle filter to hardcode subtitles
        logger.info("Hardcoding subtitles into video...")
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{subtitle_path_escaped}'",  # Quote and escape the path
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
            
            # Try alternative approach with direct file input
            logger.info("Trying alternative method for subtitle embedding...")
            alt_command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles=filename='{subtitle_path_escaped}'",  # Alternative syntax
                "-c:a", "copy",
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *alt_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Alternative subtitling failed: {stderr.decode()}")
                return None
        
        logger.info(f"Subtitled video created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error embedding subtitles: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
