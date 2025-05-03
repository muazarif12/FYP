# import os
# import re
# import asyncio
# import subprocess
# import random
# from logger_config import logger
# from ffmpeg_check import verify_dependencies

# async def generate_podcast_audio_with_gtts(podcast_data, output_dir, language="en"):
#     """
#     Generate podcast audio using gTTS (Google Text-to-Speech) without requiring pydub.
#     This function generates individual MP3 files and then uses ffmpeg to combine them.
#     Uses different TTS engines for different speakers to create a more diverse podcast.
    
#     Args:
#         podcast_data: Dictionary with podcast script information
#         output_dir: Directory to save the audio files
#         language: Language code for TTS
        
#     Returns:
#         Path to the final podcast audio file
#     """
#     # Verify dependencies first
#     deps_ok, deps_message = verify_dependencies()
#     if not deps_ok:
#         logger.error(f"Dependency check failed: {deps_message}")
#         print("\n" + deps_message)
#         return None
        
#     try:
#         # Import gtts here to avoid global import issues
#         from gtts import gTTS
        
#         # Try to import other TTS engines if available
#         try:
#             import edge_tts
#             edge_tts_available = True
#             logger.info("Edge TTS is available - will use for voice differentiation")
#         except ImportError:
#             edge_tts_available = False
#             logger.info("Edge TTS not available - falling back to gTTS with accent variation")
        
#         logger.info("Starting podcast audio generation with multiple voices...")
        
#         # Create podcast directory if it doesn't exist
#         podcast_dir = os.path.join(output_dir, "podcast")
#         os.makedirs(podcast_dir, exist_ok=True)
        
#         # Create temp directory for individual audio clips
#         temp_dir = os.path.join(podcast_dir, "temp_audio")
#         os.makedirs(temp_dir, exist_ok=True)
        
#         # Sanitize podcast title for filename
#         title = podcast_data.get('title', 'podcast')
#         safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
#         # Get script information
#         hosts = podcast_data.get('hosts', ['Host1', 'Host2'])
#         script = podcast_data.get('script', [])
        
#         if not script:
#             logger.error("No script content found")
#             return None
        
#        # Define voice settings for different speakers
#         if edge_tts_available:
#             # Use Edge TTS for better voice differentiation
#             voice_settings = {
#                 hosts[0]: {
#                     'engine': 'edge',
#                     'voice': 'en-US-GuyNeural',  # Working male voice
#                     'rate': "+0%",               # Normal rate to avoid issues
#                     'pitch': "+0Hz"              # Normal pitch
#                 },
#                 hosts[1]: {
#                     'engine': 'edge',
#                     'voice': 'en-US-JennyNeural', # Working female voice
#                     'rate': "+0%",                # Normal rate to avoid issues
#                     'pitch': "+0Hz"               # Normal pitch
#                 }
#             }
#         else:
#             # Use gTTS with different language accents for some variation
#             voice_settings = {
#                 hosts[0]: {
#                     'engine': 'gtts',
#                     'lang': language,
#                     'tld': 'com'  # US English
#                 },
#                 hosts[1]: {
#                     'engine': 'gtts',
#                     'lang': language,
#                     'tld': 'co.uk'  # UK English
#                 }
#             }
        
#         # Generate file list for ffmpeg
#         segment_files = []
        
#         logger.info(f"Generating audio for {len(script)} lines of dialogue...")
        
#         # Generate audio for each line
#         for i, line in enumerate(script):
#             speaker = line.get('speaker', hosts[i % 2])
#             text = line.get('text', '')
            
#             if not text.strip():
#                 continue
            
#             # Generate audio file for this line
#             segment_path_raw = os.path.join(temp_dir, f"line_{i:03d}_raw.mp3")
#             segment_path = os.path.join(temp_dir, f"line_{i:03d}.mp3")
            
#             # Add small pause between speakers (silent audio file)
#             if i > 0:
#                 pause_path = os.path.join(temp_dir, f"pause_{i:03d}.mp3")
#                 # Create a very short silent MP3 (will use ffmpeg)
#                 await create_silent_audio(pause_path, 0.5)  # 0.7 seconds of silence
#                 segment_files.append(pause_path)
            
#             # Get voice settings for this speaker
#             voice = voice_settings.get(speaker, voice_settings[hosts[0]])
            
#             # Generate speech with the appropriate engine
#             try:
#                 if voice['engine'] == 'edge' and edge_tts_available:
#                     # Use Edge TTS with specific rate and pitch
#                     await generate_edge_tts(
#                         text, 
#                         segment_path_raw, 
#                         voice['voice'],
#                         voice.get('rate', '+0%'),
#                         voice.get('pitch', '+0Hz')
#                     )
#                 else:
#                     # Fallback to gTTS
#                     tts = gTTS(text=text, lang=voice['lang'], tld=voice.get('tld', 'com'), slow=False)
#                     tts.save(segment_path_raw)
                
#                 # Normalize audio for consistent volume
#                 await normalize_audio(segment_path_raw, segment_path)
#                 segment_files.append(segment_path)
                
#                 # Clean up raw file
#                 try:
#                     os.remove(segment_path_raw)
#                 except:
#                     pass
                
#                 logger.info(f"Generated audio segment {i+1}/{len(script)}: {speaker}")
#             except Exception as e:
#                 logger.error(f"Error generating audio for segment {i+1}: {e}")
#                 # Create empty file to avoid breaking the chain
#                 await create_silent_audio(segment_path, 1.0)
#                 segment_files.append(segment_path)
        
#         # Create file list for ffmpeg
#         file_list_path = os.path.join(temp_dir, "file_list.txt")
#         with open(file_list_path, 'w', encoding='utf-8') as f:
#             for file_path in segment_files:
#                 f.write(f"file '{os.path.basename(file_path)}'\n")
        
#         # Output path for final audio
#         final_path = os.path.join(podcast_dir, f"{safe_title}.mp3")
#         final_raw_path = os.path.join(podcast_dir, f"{safe_title}_raw.mp3")
        
#         # Use ffmpeg to concatenate all files
#         logger.info("Combining audio segments with ffmpeg...")
        
#         try:
#             # Check if ffmpeg is available
#             ffmpeg_command = [
#                 "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
#                 "-i", file_list_path, "-c", "copy", final_raw_path
#             ]
            
#             # Run ffmpeg command
#             process = await asyncio.create_subprocess_exec(
#                 *ffmpeg_command,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE,
#                 cwd=temp_dir  # Set working directory to temp_dir so relative paths work
#             )
            
#             stdout, stderr = await process.communicate()
            
#             if process.returncode != 0:
#                 logger.error(f"FFmpeg error: {stderr.decode()}")
#                 # Try alternative approach if ffmpeg fails
#                 return await manual_audio_combine(segment_files, final_path)
            
#             # Apply final audio mastering
#             await master_audio(final_raw_path, final_path)
            
#             logger.info(f"Podcast audio successfully saved to: {final_path}")
            
#             # Clean up temp files if successful
#             for file_path in segment_files:
#                 try:
#                     os.remove(file_path)
#                 except:
#                     pass
#             try:
#                 os.remove(file_list_path)
#                 os.remove(final_raw_path)
#                 os.rmdir(temp_dir)
#             except:
#                 pass
                
#             return final_path
            
#         except Exception as e:
#             logger.error(f"Error combining audio: {e}")
#             # Try alternative approach if ffmpeg fails
#             return await manual_audio_combine(segment_files, final_path)
        
#     except ImportError as e:
#         logger.error(f"Required library not found: {e}. Install with 'pip install gtts'")
#         return None
#     except Exception as e:
#         logger.error(f"Error in audio generation: {e}")
#         return None

# async def generate_edge_tts(text, output_path, voice="en-US-GuyNeural", rate="+0%", pitch="+0Hz"):
#     """Generate audio using Microsoft Edge TTS with consistent rate and pitch."""
#     try:
#         import edge_tts
        
#         # List of available voices (fallback if the requested voice has issues)
#         male_voices = ["en-US-GuyNeural", "en-US-JasonNeural", "en-US-TonyNeural"]
#         female_voices = ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-MichelleNeural"]
        
#         # If the voice contains "Christopher" or "Sara", use a working alternative
#         if "Christopher" in voice:
#             voice = "en-US-GuyNeural"  # Use the standard Guy voice instead
#         elif "Sara" in voice:
#             voice = "en-US-JennyNeural"  # Use the standard Jenny voice instead
            
#         # Fix rate parameter to ensure it's valid
#         # Edge TTS accepts rates from -100% to +100%
#         if rate != "+0%":
#             # Ensure rate is within valid range and formatted correctly
#             try:
#                 rate_value = int(rate.replace('%', '').replace('+', '').replace('-', ''))
#                 rate_sign = '-' if '-' in rate else '+'
#                 rate_value = min(100, max(0, rate_value))  # Clamp between 0-100
#                 rate = f"{rate_sign}{rate_value}%"
#             except:
#                 rate = "+0%"  # Default to normal rate if parsing fails
        
#         logger.info(f"Using Edge TTS with voice: {voice}, rate: {rate}")
        
#         # Add rate and pitch parameters for more consistent sound
#         communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
#         await communicate.save(output_path)
#         return output_path
#     except Exception as e:
#         # Fallback to gTTS if edge_tts fails
#         logger.error(f"Edge TTS failed: {e}. Falling back to gTTS")
#         from gtts import gTTS
#         tts = gTTS(text=text, lang='en', slow=False)
#         tts.save(output_path)
#         return output_path

# async def create_silent_audio(output_path, duration=0.5):
#     """Create a silent audio file using ffmpeg."""
#     try:
#         # Use ffmpeg to create silent audio
#         ffmpeg_command = [
#             "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono", 
#             "-t", str(duration), "-q:a", "0", "-c:a", "libmp3lame", output_path
#         ]
        
#         process = await asyncio.create_subprocess_exec(
#             *ffmpeg_command,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         await process.communicate()
        
#         if process.returncode != 0 or not os.path.exists(output_path):
#             # Create an empty file if ffmpeg fails
#             with open(output_path, 'wb') as f:
#                 # Write minimal valid MP3 header
#                 f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        
#         return output_path
#     except Exception as e:
#         logger.error(f"Error creating silent audio: {e}")
#         # Create an empty file
#         with open(output_path, 'wb') as f:
#             f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
#         return output_path

# async def normalize_audio(input_path, output_path):
#     """Normalize audio volume for more consistent sound."""
#     try:
#         norm_command = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-filter:a", "loudnorm=I=-16:LRA=11:TP=-1.5",
#             "-ar", "44100", 
#             output_path
#         ]
        
#         process = await asyncio.create_subprocess_exec(
#             *norm_command,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         await process.communicate()
        
#         if os.path.exists(output_path):
#             return output_path
#         return input_path  # Return original if normalization fails
#     except Exception as e:
#         logger.error(f"Audio normalization failed: {e}")
#         return input_path  # Return original if normalization fails

# async def master_audio(input_path, output_path):
#     """Apply audio mastering to the final podcast for professional sound."""
#     try:
#         master_command = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-af", "equalizer=f=1000:width_type=o:width=200:g=-3,equalizer=f=3000:width_type=o:width=200:g=2,loudnorm=I=-16:LRA=11:TP=-1.5,acompressor=threshold=-8dB:ratio=4:attack=200:release=1000",
#             "-ar", "44100", "-b:a", "192k",
#             output_path
#         ]
        
#         process = await asyncio.create_subprocess_exec(
#             *master_command,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         await process.communicate()
        
#         if os.path.exists(output_path):
#             logger.info(f"Audio mastering successful: {output_path}")
#             return output_path
#         return input_path  # Return original if mastering fails
#     except Exception as e:
#         logger.error(f"Audio mastering failed: {e}")
#         return input_path  # Return original if mastering fails

# async def manual_audio_combine(segment_files, output_path):
#     """
#     Fallback method to combine audio files if ffmpeg concatenation fails.
#     This uses a simpler ffmpeg command that should work in most environments.
#     """
#     try:
#         logger.info("Using alternative method to combine audio files...")
        
#         # Create a temporary file with all content
#         temp_file = output_path + ".temp.mp3"
        
#         # Copy the first file as a starting point
#         if segment_files and os.path.exists(segment_files[0]):
#             import shutil
#             shutil.copy(segment_files[0], temp_file)
            
#             # For each remaining file, append it to the temp file
#             for i, file_path in enumerate(segment_files[1:], 1):
#                 if not os.path.exists(file_path):
#                     continue
                    
#                 # Output file for this iteration
#                 output_temp = f"{output_path}.{i}.mp3"
                
#                 # Use ffmpeg to concatenate this segment
#                 cmd = [
#                     "ffmpeg", "-y",
#                     "-i", temp_file,
#                     "-i", file_path,
#                     "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
#                     "-map", "[out]",
#                     output_temp
#                 ]
                
#                 process = await asyncio.create_subprocess_exec(
#                     *cmd,
#                     stdout=asyncio.subprocess.PIPE,
#                     stderr=asyncio.subprocess.PIPE
#                 )
                
#                 await process.communicate()
                
#                 # If successful, replace the temp file with this iteration's output
#                 if os.path.exists(output_temp):
#                     try:
#                         os.remove(temp_file)
#                     except:
#                         pass
#                     os.rename(output_temp, temp_file)
        
#         # Apply mastering to the final combined file
#         final_mastered_path = output_path
#         await master_audio(temp_file, final_mastered_path)
        
#         # Rename the temp file to the final output
#         try:
#             if os.path.exists(final_mastered_path):
#                 logger.info(f"Podcast audio saved to: {final_mastered_path}")
#                 os.remove(temp_file)
#                 return final_mastered_path
#             else:
#                 os.rename(temp_file, output_path)
#                 logger.info(f"Podcast audio saved to: {output_path}")
#                 return output_path
#         except Exception as e:
#             logger.error(f"Error finalizing audio: {e}")
#             return temp_file
            
#     except Exception as e:
#         logger.error(f"Error in manual audio combine: {e}")
#         return None


import os
import re
import asyncio
import shutil
import tempfile
import subprocess
from logger_config import logger
from ffmpeg_check import verify_dependencies

async def generate_podcast_audio_with_gtts(podcast_data, output_dir, language="en"):
    """
    Simplified podcast audio generator that focuses on reliability over features.
    Uses gTTS for all voice generation and basic ffmpeg operations.
    """
    # Verify dependencies first
    deps_ok, deps_message = verify_dependencies()
    if not deps_ok:
        logger.error(f"Dependency check failed: {deps_message}")
        print("\n" + deps_message)
        return None
        
    temp_dir = None
    
    try:
        # Import gtts here to avoid global import issues
        from gtts import gTTS
        
        # Check for Edge TTS
        edge_tts_available = False
        try:
            import edge_tts
            edge_tts_available = True
            logger.info("Edge TTS is available - will use for voice differentiation")
        except ImportError:
            logger.info("Edge TTS not available - using only gTTS")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        # Create podcast directory for final output
        podcast_dir = os.path.join(output_dir, "podcast")
        os.makedirs(podcast_dir, exist_ok=True)
        
        # Sanitize podcast title for filename
        title = podcast_data.get('title', 'podcast')
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
        # Get script information
        hosts = podcast_data.get('hosts', ['Host1', 'Host2'])
        script = podcast_data.get('script', [])
        
        if not script:
            logger.error("No script content found")
            return None
        
        # Define voice settings based on available engines
        if edge_tts_available:
            # Simple Edge TTS configuration - focus on reliability
            voice_options = {
                hosts[0]: {
                    'engine': 'edge', 
                    'voice': 'en-US-GuyNeural'  # Male voice
                },
                hosts[1]: {
                    'engine': 'edge',
                    'voice': 'en-US-JennyNeural'  # Female voice
                }
            }
        else:
            # gTTS fallback with different accents
            voice_options = {
                hosts[0]: {
                    'engine': 'gtts',
                    'lang': language,
                    'tld': 'com'  # US accent
                },
                hosts[1]: {
                    'engine': 'gtts',
                    'lang': language,
                    'tld': 'co.uk'  # UK accent
                }
            }
        
        # Generate audio segments
        logger.info(f"Generating audio for {len(script)} lines of dialogue...")
        segment_files = []
        
        for i, line in enumerate(script):
            speaker = line.get('speaker', hosts[i % len(hosts)])
            text = line.get('text', '').strip()
            
            if not text:
                continue
                
            # Define output path for this segment
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
            
            # Get voice settings for this speaker
            voice = voice_options.get(speaker, voice_options[hosts[0]])
            
            # Generate audio based on engine type
            success = False
            
            try:
                if voice['engine'] == 'edge' and edge_tts_available:
                    # Try Edge TTS first
                    try:
                        communicate = edge_tts.Communicate(text, voice['voice'])
                        await communicate.save(segment_path)
                        success = os.path.exists(segment_path) and os.path.getsize(segment_path) > 0
                    except Exception as e:
                        logger.error(f"Edge TTS failed for segment {i+1}: {str(e)}")
                        success = False
                
                # Fallback to gTTS if Edge TTS fails or is not selected
                if not success:
                    if voice['engine'] == 'gtts' or not success:
                        lang = voice.get('lang', language)
                        tld = voice.get('tld', 'com')
                        
                        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
                        tts.save(segment_path)
                        success = os.path.exists(segment_path) and os.path.getsize(segment_path) > 0
                
                if success:
                    segment_files.append(segment_path)
                    logger.info(f"Generated audio segment {i+1}/{len(script)}: {speaker}")
                else:
                    logger.error(f"Failed to generate audio for segment {i+1}")
            except Exception as e:
                logger.error(f"Error generating audio for segment {i+1}: {str(e)}")
        
        # Check if we have any segments to combine
        if not segment_files:
            logger.error("No audio segments were generated successfully")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None
            
        # Output path for final audio
        final_path = os.path.join(podcast_dir, f"{safe_title}.mp3")
        
        # If only one segment, just copy it
        if len(segment_files) == 1:
            shutil.copy(segment_files[0], final_path)
            logger.info(f"Only one segment - copied directly to: {final_path}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return final_path
        
        # Create combined audio
        combined = await combine_audio_segments(segment_files, final_path, temp_dir)
        
        if combined and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            logger.info(f"Podcast audio successfully saved to: {final_path}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return final_path
        else:
            # If all methods fail, use the first segment as a fallback
            logger.warning("All audio combination methods failed - using first segment as output")
            try:
                shutil.copy(segment_files[0], final_path)
                logger.info(f"Used first segment as output: {final_path}")
                return final_path
            except Exception as copy_error:
                logger.error(f"Failed to copy first segment: {str(copy_error)}")
                return None
                
    except Exception as e:
        logger.error(f"Error in podcast audio generation: {str(e)}")
        return None
    finally:
        # Make sure temp directory is cleaned up
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up temp files: {str(cleanup_error)}")


async def combine_audio_segments(segment_files, output_path, temp_dir):
    """
    Try multiple methods to combine audio segments until one works.
    Returns True if successful, False otherwise.
    """
    # Method 1: Use FFmpeg with direct input files (most reliable method)
    try:
        logger.info("Trying direct FFmpeg concatenation method")
        
        # Build FFmpeg command with all input files and concat filter
        cmd = ["ffmpeg", "-y"]
        
        # Add input files
        for file in segment_files:
            cmd.extend(["-i", file])
        
        # Create filter_complex argument
        filter_complex = ""
        for i in range(len(segment_files)):
            filter_complex += f"[{i}:a:0]"
        filter_complex += f"concat=n={len(segment_files)}:v=0:a=1[out]"
        
        # Complete command
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[out]",
            output_path
        ])
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Direct FFmpeg concatenation successful")
            return True
        else:
            logger.error(f"Direct FFmpeg failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error in direct FFmpeg method: {str(e)}")
    
    # Method 2: Use FFmpeg with concat demuxer
    try:
        logger.info("Trying FFmpeg concat demuxer method")
        
        # Create file list
        list_file = os.path.join(temp_dir, "filelist.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for file in segment_files:
                # Make sure to escape single quotes in filenames
                escaped_path = file.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # Create FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("FFmpeg concat demuxer successful")
            return True
        else:
            logger.error(f"FFmpeg concat demuxer failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error in concat demuxer method: {str(e)}")
    
    # Method 3: Use binary concatenation for compatible MP3 files
    try:
        logger.info("Trying binary concatenation method")
        
        if os.name == 'posix':  # Linux/Mac
            # Use cat command
            with open(output_path, 'wb') as outfile:
                for file in segment_files:
                    with open(file, 'rb') as infile:
                        outfile.write(infile.read())
        else:  # Windows
            # Use Windows copy command
            combined_file = os.path.join(temp_dir, "combined.mp3")
            
            # Write a batch file to do the concatenation
            bat_file = os.path.join(temp_dir, "concat.bat")
            with open(bat_file, 'w') as f:
                files_str = " + ".join([f'"{file}"' for file in segment_files])
                f.write(f'copy /b {files_str} "{combined_file}"\n')
            
            # Execute the batch file
            subprocess.run(
                ['cmd.exe', '/c', bat_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Check if combined file was created and copy it to output
            if os.path.exists(combined_file) and os.path.getsize(combined_file) > 0:
                shutil.copy(combined_file, output_path)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Binary concatenation successful")
            return True
        else:
            logger.error("Binary concatenation failed")
    except Exception as e:
        logger.error(f"Error in binary concatenation method: {str(e)}")
    
    # Method 4: Use an mp3 specific joiner (MP3wrap or similar) if available
    try:
        logger.info("Trying MP3wrap or similar tool if available")
        
        # Check if MP3wrap is available
        mp3wrap_available = False
        try:
            subprocess.run(
                ["mp3wrap", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            mp3wrap_available = True
        except:
            mp3wrap_available = False
        
        if mp3wrap_available:
            # Use MP3wrap to join files
            temp_output = os.path.join(temp_dir, "output_wrap.mp3")
            cmd = ["mp3wrap", temp_output] + segment_files
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # MP3wrap adds a prefix to the filename
            wrapped_file = temp_output.replace(".mp3", "_MP3WRAP.mp3")
            if os.path.exists(wrapped_file):
                shutil.copy(wrapped_file, output_path)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("MP3wrap concatenation successful")
            return True
    except Exception as e:
        logger.error(f"Error in MP3wrap method: {str(e)}")
    
    # Try one final, extremely simple approach for Windows
    if os.name != 'posix':
        try:
            logger.info("Trying simple Windows CMD copy approach")
            copy_cmd = 'copy /b '
            for i, file in enumerate(segment_files):
                if i > 0:
                    copy_cmd += '+'
                copy_cmd += f'"{file}" '
            copy_cmd += f'"{output_path}"'
            
            # Execute the command directly
            os.system(copy_cmd)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("Simple Windows copy concatenation successful")
                return True
        except Exception as e:
            logger.error(f"Error in simple Windows copy approach: {str(e)}")
    
    logger.error("All concatenation methods failed")
    return False