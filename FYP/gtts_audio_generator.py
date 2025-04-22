import os
import re
import asyncio
import subprocess
from logger_config import logger
from ffmpeg_check import verify_dependencies

async def generate_podcast_audio_with_gtts(podcast_data, output_dir, language="en"):
    """
    Generate podcast audio using gTTS (Google Text-to-Speech) without requiring pydub.
    This function generates individual MP3 files and then uses ffmpeg to combine them.
    
    Args:
        podcast_data: Dictionary with podcast script information
        output_dir: Directory to save the audio files
        language: Language code for TTS
        
    Returns:
        Path to the final podcast audio file
    """
    # Verify dependencies first
    deps_ok, deps_message = verify_dependencies()
    if not deps_ok:
        logger.error(f"Dependency check failed: {deps_message}")
        print("\n" + deps_message)
        return None
        
    try:
        # Import gtts here to avoid global import issues
        from gtts import gTTS
        
        logger.info("Starting podcast audio generation with gTTS...")
        
        # Create podcast directory if it doesn't exist
        podcast_dir = os.path.join(output_dir, "podcast")
        os.makedirs(podcast_dir, exist_ok=True)
        
        # Create temp directory for individual audio clips
        temp_dir = os.path.join(podcast_dir, "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sanitize podcast title for filename
        title = podcast_data.get('title', 'podcast')
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
        # Get script information
        hosts = podcast_data.get('hosts', ['Host1', 'Host2'])
        script = podcast_data.get('script', [])
        
        if not script:
            logger.error("No script content found")
            return None
        
        # Generate file list for ffmpeg
        file_list_path = os.path.join(temp_dir, "file_list.txt")
        segment_files = []
        
        logger.info(f"Generating audio for {len(script)} lines of dialogue...")
        
        # Generate audio for each line
        for i, line in enumerate(script):
            speaker = line.get('speaker', hosts[i % 2])
            text = line.get('text', '')
            
            if not text.strip():
                continue
            
            # Generate audio file for this line
            segment_path = os.path.join(temp_dir, f"line_{i:03d}.mp3")
            segment_files.append(segment_path)
            
            # Add small pause between speakers (silent audio file)
            if i > 0:
                pause_path = os.path.join(temp_dir, f"pause_{i:03d}.mp3")
                # Create a very short silent MP3 (will use ffmpeg)
                await create_silent_audio(pause_path, 0.7)  # 0.7 seconds of silence
                segment_files.append(pause_path)
            
            # Generate speech with gTTS
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(segment_path)
                logger.info(f"Generated audio segment {i+1}/{len(script)}: {speaker}")
            except Exception as e:
                logger.error(f"Error generating audio for segment {i+1}: {e}")
                # Create empty file to avoid breaking the chain
                await create_silent_audio(segment_path, 1.0)
        
        # Create file list for ffmpeg
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for file_path in segment_files:
                f.write(f"file '{os.path.basename(file_path)}'\n")
        
        # Output path for final audio
        final_path = os.path.join(podcast_dir, f"{safe_title}.mp3")
        
        # Use ffmpeg to concatenate all files
        logger.info("Combining audio segments with ffmpeg...")
        
        try:
            # Check if ffmpeg is available
            ffmpeg_command = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                "-i", file_list_path, "-c", "copy", final_path
            ]
            
            # Run ffmpeg command
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir  # Set working directory to temp_dir so relative paths work
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                # Try alternative approach if ffmpeg fails
                return await manual_audio_combine(segment_files, final_path)
            
            logger.info(f"Podcast audio successfully saved to: {final_path}")
            
            # Clean up temp files if successful
            for file_path in segment_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                os.remove(file_list_path)
                os.rmdir(temp_dir)
            except:
                pass
                
            return final_path
            
        except Exception as e:
            logger.error(f"Error combining audio: {e}")
            # Try alternative approach if ffmpeg fails
            return await manual_audio_combine(segment_files, final_path)
        
    except ImportError as e:
        logger.error(f"Required library not found: {e}. Install with 'pip install gtts'")
        return None
    except Exception as e:
        logger.error(f"Error in audio generation: {e}")
        return None

async def create_silent_audio(output_path, duration=0.5):
    """Create a silent audio file using ffmpeg."""
    try:
        # Use ffmpeg to create silent audio
        ffmpeg_command = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono", 
            "-t", str(duration), "-q:a", "0", "-c:a", "libmp3lame", output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0 or not os.path.exists(output_path):
            # Create an empty file if ffmpeg fails
            with open(output_path, 'wb') as f:
                # Write minimal valid MP3 header
                f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        
        return output_path
    except Exception as e:
        logger.error(f"Error creating silent audio: {e}")
        # Create an empty file
        with open(output_path, 'wb') as f:
            f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        return output_path

async def manual_audio_combine(segment_files, output_path):
    """
    Fallback method to combine audio files if ffmpeg concatenation fails.
    This uses a simpler ffmpeg command that should work in most environments.
    """
    try:
        logger.info("Using alternative method to combine audio files...")
        
        # Create a temporary file with all content
        temp_file = output_path + ".temp.mp3"
        
        # Use ffmpeg to append files one by one
        for i, file_path in enumerate(segment_files):
            if i == 0:
                # For first file, just copy it
                cmd = ["ffmpeg", "-y", "-i", file_path, "-c", "copy", temp_file]
            else:
                # For subsequent files, append to the temp file
                cmd = [
                    "ffmpeg", "-y", 
                    "-i", temp_file, "-i", file_path,
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]", 
                    "-map", "[out]", output_path + ".new.mp3"
                ]
                
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if i > 0 and process.returncode == 0:
                # Rename the new file to the temp file for the next iteration
                try:
                    os.remove(temp_file)
                except:
                    pass
                os.rename(output_path + ".new.mp3", temp_file)
        
        # Rename the temp file to the final output
        try:
            os.rename(temp_file, output_path)
            logger.info(f"Podcast audio saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error finalizing audio: {e}")
            return temp_file
            
    except Exception as e:
        logger.error(f"Error in manual audio combine: {e}")
        return None