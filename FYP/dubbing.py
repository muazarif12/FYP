# import os
# import asyncio
# import time
# import re
# from logger_config import logger
# from ffmpeg_check import verify_dependencies
# # from gtts_audio_generator import generate_edge_tts
# from utils import format_timestamp
# import ollama

# async def translate_transcript_to_english(transcript_segments, detected_language):
#     """
#     Translate the transcript segments to English using Deep Translator.
    
#     Args:
#         transcript_segments: List of tuples (start_time, end_time, text)
#         detected_language: The detected language code
        
#     Returns:
#         List of tuples (start_time, end_time, translated_text)
#     """
#     if detected_language == "en":
#         logger.info("Transcript already in English, no translation needed")
#         return transcript_segments
    
#     try:
#         logger.info(f"Translating transcript from {detected_language} to English using Deep Translator...")
        
#         # Import the translation library
#         try:
#             from deep_translator import GoogleTranslator
#             translator = GoogleTranslator(source=detected_language, target='en')
#         except ImportError:
#             logger.warning("deep_translator not installed. Install with: pip install deep_translator")
#             logger.warning("Falling back to Ollama for translation")
#             return await translate_transcript_with_ollama(transcript_segments, detected_language)
        
#         translated_segments = []
#         batch_size = 25  # Deep translator recommends smaller batches
        
#         # Process in batches for speed and to avoid rate limits
#         for i in range(0, len(transcript_segments), batch_size):
#             batch = transcript_segments[i:i+batch_size]
            
#             # Process each segment in the batch
#             for j, (start, end, text) in enumerate(batch):
#                 try:
#                     # Skip empty or very short segments
#                     if not text or len(text.strip()) < 2:
#                         translated_segments.append((start, end, text))
#                         continue
                    
#                     # Translate the segment
#                     translated_text = translator.translate(text)
#                     translated_segments.append((start, end, translated_text))
                    
#                     # Log progress periodically
#                     if (i + j) % 50 == 0:
#                         logger.info(f"Translated {i + j}/{len(transcript_segments)} segments")
#                 except Exception as e:
#                     logger.error(f"Error translating segment {i+j}: {e}")
#                     # Keep original text as fallback
#                     translated_segments.append((start, end, text))
            
#             # Small delay to avoid rate limits
#             await asyncio.sleep(0.2)
        
#         logger.info(f"Translation completed. Translated {len(translated_segments)} segments.")
#         return translated_segments
    
#     except Exception as e:
#         logger.error(f"Error in Deep Translator: {e}")
#         # Fall back to Ollama translation if Google fails
#         logger.info("Falling back to Ollama for translation")
#         return await translate_transcript_with_ollama(transcript_segments, detected_language)

# async def translate_transcript_with_ollama(transcript_segments, detected_language):
#     """
#     Fallback translation using Ollama LLM.
#     """
#     try:
#         logger.info(f"Translating transcript from {detected_language} to English using Ollama...")
        
#         # Prepare chunks of transcript for translation (to avoid token limits)
#         chunk_size = 20  # Number of segments per chunk
#         chunks = [transcript_segments[i:i + chunk_size] for i in range(0, len(transcript_segments), chunk_size)]
        
#         translated_segments = []
        
#         for i, chunk in enumerate(chunks):
#             # Extract text from segments
#             text_list = [seg[2] for seg in chunk]
#             text_with_indices = "\n".join([f"[{idx}] {text}" for idx, text in enumerate(text_list)])
            
#             # Create translation prompt
#             prompt = f"""
#             Translate the following text segments from {detected_language} to English.
#             Maintain the original meaning and tone. Keep all numbers, names, and technical terms intact.
#             The translations should be natural and fluent English, not literal translations.
            
#             Return only the translations with their index numbers in this exact format:
#             [0] English translation of first segment
#             [1] English translation of second segment
#             ...and so on.
            
#             Text to translate:
#             {text_with_indices}
#             """
            
#             # Call Ollama for translation
#             logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} segments)")
#             response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "system", "content": prompt}])
#             result = response["message"]["content"]
            
#             # Parse the translations
#             translations = {}
#             for line in result.split('\n'):
#                 line = line.strip()
#                 if not line:
#                     continue
                    
#                 match = re.match(r'\[(\d+)\](.*)', line)
#                 if match:
#                     index = int(match.group(1))
#                     translated_text = match.group(2).strip()
#                     translations[index] = translated_text
            
#             # Map translations back to original segments with timing intact
#             for j, (start, end, _) in enumerate(chunk):
#                 translated_text = translations.get(j, "")
#                 if not translated_text:
#                     # Fallback if translation is missing
#                     translated_text = chunk[j][2]
#                     logger.warning(f"Missing translation for segment {j} in chunk {i}, using original")
                
#                 translated_segments.append((start, end, translated_text))
            
#             # Add a small delay to avoid rate limits
#             await asyncio.sleep(0.5)
        
#         logger.info(f"Translation completed. Translated {len(translated_segments)} segments.")
#         return translated_segments
    
#     except Exception as e:
#         logger.error(f"Error translating with Ollama: {e}")
#         # Return original if translation fails
#         return transcript_segments

# # Update the generate_dubbed_audio function in dubbing.py

# # Update the generate_dubbed_audio function to use only one voice per gender

# async def generate_dubbed_audio(translated_segments, output_dir, video_duration, gender_predictions=None):
#     """
#     Generate English audio for each translated segment with one consistent voice per gender.
    
#     Args:
#         translated_segments: List of tuples (start_time, end_time, translated_text)
#         output_dir: Directory to save audio files
#         video_duration: Duration of the video in seconds
#         gender_predictions: Dictionary mapping segment indices to predicted gender
        
#     Returns:
#         Path to the directory containing audio segments
#     """
#     try:
#         deps_ok, deps_message = verify_dependencies()
#         if not deps_ok:
#             logger.error(f"Dependency check failed: {deps_message}")
#             return None
            
#         # Create temp directory for audio segments
#         temp_dir = os.path.join(output_dir, "dubbing", "temp_audio")
#         os.makedirs(temp_dir, exist_ok=True)
        
#         # Set up one voice per gender - use the most reliable voices
#         male_voice = "en-US-GuyNeural"    # Single consistent male voice
#         female_voice = "en-US-JennyNeural" # Single consistent female voice
        
#         # Default to alternating genders if no gender predictions
#         if not gender_predictions:
#             logger.info("No gender predictions provided. Using consistent male/female voices based on segment index.")
#             gender_predictions = {i: 'male' if i % 2 == 0 else 'female' for i in range(len(translated_segments))}
        
#         logger.info(f"Generating English audio for {len(translated_segments)} segments with consistent gender voices...")
#         logger.info(f"Using '{male_voice}' for all male segments and '{female_voice}' for all female segments")
        
#         # Track the progress
#         total_segments = len(translated_segments)
#         audio_segment_paths = []
        
#         # Try using Edge TTS if available, with fallback to gTTS
#         try:
#             import edge_tts
#             edge_tts_available = True
#             logger.info("Using Edge TTS for gender-appropriate voices")
#         except ImportError:
#             edge_tts_available = False
#             logger.info("Edge TTS not available - falling back to gTTS (without gender distinction)")
#             from gtts import gTTS
        
#         # Generate audio for each segment
#         for i, (start, end, text) in enumerate(translated_segments):
#             segment_duration = end - start
#             segment_path = os.path.join(temp_dir, f"segment_{i:04d}_{start:.2f}_{end:.2f}.mp3")
            
#             # Skip empty segments
#             if not text.strip():
#                 logger.info(f"Skipping empty segment {i}/{total_segments}")
#                 continue
            
#             try:
#                 if i % 10 == 0:  # Log progress every 10 segments
#                     logger.info(f"Generating audio segment {i+1}/{total_segments}")
                
#                 # Determine gender for this segment
#                 gender = gender_predictions.get(i, 'male')  # Default to male if not specified
                
#                 # Select voice based on gender - only one voice per gender
#                 selected_voice = male_voice if gender == 'male' else female_voice
                
#                 if edge_tts_available:
#                     try:
#                         # Generate with Edge TTS
#                         logger.info(f"Using {gender} voice: {selected_voice}")
#                         communicate = edge_tts.Communicate(text, selected_voice)
#                         await communicate.save(segment_path)
#                     except Exception as e:
#                         logger.error(f"Edge TTS failed: {e}. Falling back to gTTS")
#                         # Fallback to gTTS
#                         tts = gTTS(text=text, lang='en', slow=False)
#                         tts.save(segment_path)
#                 else:
#                     # Use gTTS (no gender distinction)
#                     tts = gTTS(text=text, lang='en', slow=False)
#                     tts.save(segment_path)
                
#                 # Store the path along with timing information
#                 if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
#                     audio_segment_paths.append({
#                         "path": segment_path,
#                         "start": start,
#                         "end": end,
#                         "text": text,
#                         "gender": gender
#                     })
#                 else:
#                     logger.warning(f"Segment {i} audio file is empty or not created")
#             except Exception as e:
#                 logger.error(f"Error generating audio for segment {i}: {e}")
#                 # Try with a shorter text if original fails
#                 try:
#                     shortened_text = text[:200] + "..." if len(text) > 200 else text
#                     tts = gTTS(text=shortened_text, lang='en', slow=False)
#                     tts.save(segment_path)
                    
#                     if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
#                         audio_segment_paths.append({
#                             "path": segment_path,
#                             "start": start,
#                             "end": end,
#                             "text": shortened_text,
#                             "gender": gender
#                         })
#                 except Exception as inner_e:
#                     logger.error(f"Error with shortened text for segment {i}: {inner_e}")
        
#         logger.info(f"Generated {len(audio_segment_paths)} audio segments")
        
#         # Create a manifest file with timing information
#         manifest_path = os.path.join(temp_dir, "manifest.json")
#         with open(manifest_path, 'w', encoding='utf-8') as f:
#             json.dump({
#                 "segments": audio_segment_paths,
#                 "video_duration": video_duration,
#                 "voice_settings": {
#                     "male_voice": male_voice,
#                     "female_voice": female_voice
#                 }
#             }, f, indent=2)
        
#         return temp_dir
    
#     except Exception as e:
#         logger.error(f"Error generating dubbed audio: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None

# # Update the synchronize_dubbing function in dubbing.py

# async def synchronize_dubbing(video_path, audio_segments_dir, output_dir):
#     """
#     Synchronize the dubbed audio with the original video.
    
#     Args:
#         video_path: Path to the original video
#         audio_segments_dir: Directory containing audio segments
#         output_dir: Directory to save the dubbed video
        
#     Returns:
#         Path to the dubbed video
#     """
#     try:
#         # Load the audio segment manifest
#         manifest_path = os.path.join(audio_segments_dir, "manifest.json")
        
#         # Check if manifest exists
#         if not os.path.exists(manifest_path):
#             logger.error(f"Manifest file not found at {manifest_path}")
#             return None
            
#         with open(manifest_path, 'r', encoding='utf-8') as f:
#             manifest = json.load(f)
        
#         segments = manifest["segments"]
#         video_duration = manifest["video_duration"]
        
#         # Create output directory
#         dubbing_dir = os.path.join(output_dir, "dubbing")
#         os.makedirs(dubbing_dir, exist_ok=True)
        
#         # Make sure we have some segments to work with
#         if not segments:
#             logger.error("No audio segments found in manifest")
#             return None
            
#         logger.info(f"Found {len(segments)} audio segments to process")
        
#         # First, let's verify all segment files exist
#         existing_segments = []
#         for segment in segments:
#             path = segment["path"]
#             if os.path.exists(path) and os.path.getsize(path) > 0:
#                 existing_segments.append(segment)
#             else:
#                 logger.warning(f"Segment file not found or empty: {path}")
        
#         if not existing_segments:
#             logger.error("No valid audio segments found")
#             return None
            
#         logger.info(f"{len(existing_segments)} valid audio segments found")
        
#         # Create a file list for concatenation with absolute paths
#         concat_file = os.path.join(audio_segments_dir, "concat.txt")
#         with open(concat_file, 'w', encoding='utf-8') as f:
#             for segment in existing_segments:
#                 # Use absolute path to avoid directory issues
#                 abs_path = os.path.abspath(segment["path"])
#                 # Escape backslashes in Windows paths
#                 safe_path = abs_path.replace('\\', '\\\\')
#                 f.write(f"file '{safe_path}'\n")
        
#         # Output path for combined audio
#         combined_audio = os.path.join(audio_segments_dir, "combined_audio.mp3")
        
#         # Use ffmpeg to concatenate all audio files
#         concat_cmd = [
#             "ffmpeg", "-y", "-f", "concat", "-safe", "0",
#             "-i", concat_file, "-c", "copy", combined_audio
#         ]
        
#         # Run the concatenation command
#         logger.info("Combining audio segments...")
#         logger.info(f"Using concat file: {concat_file}")
        
#         # Check if concat file exists and has content
#         if os.path.exists(concat_file):
#             with open(concat_file, 'r') as f:
#                 logger.info(f"Concat file first few lines: {f.readline()[:100]}")
#         else:
#             logger.error(f"Concat file does not exist: {concat_file}")
#             return None
        
#         process = await asyncio.create_subprocess_exec(
#             *concat_cmd,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         stdout, stderr = await process.communicate()
        
#         if process.returncode != 0:
#             logger.error(f"Error concatenating audio: {stderr.decode()}")
            
#             # Try alternative approach using a different concat method
#             logger.info("Trying alternative audio concatenation method...")
            
#             # Use direct file inputs instead of a concat file
#             alt_concat_cmd = ["ffmpeg", "-y"]
            
#             # Add each file as an input
#             for segment in existing_segments[:10]:  # Limit to first 10 for testing
#                 alt_concat_cmd.extend(["-i", segment["path"]])
            
#             # Add filter complex to merge all inputs
#             filter_str = ""
#             for i in range(min(10, len(existing_segments))):
#                 filter_str += f"[{i}:0]"
#             filter_str += f"concat=n={min(10, len(existing_segments))}:v=0:a=1[out]"
            
#             alt_concat_cmd.extend([
#                 "-filter_complex", filter_str,
#                 "-map", "[out]",
#                 combined_audio
#             ])
            
#             process = await asyncio.create_subprocess_exec(
#                 *alt_concat_cmd,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )
            
#             stdout, stderr = await process.communicate()
            
#             if process.returncode != 0:
#                 logger.error(f"Alternative audio concatenation failed: {stderr.decode()}")
                
#                 # If both methods fail, try creating a minimal audio file for testing
#                 logger.info("Creating minimal audio for testing...")
#                 if len(existing_segments) > 0:
#                     import shutil
#                     try:
#                         # Just copy the first segment as the combined audio
#                         shutil.copy(existing_segments[0]["path"], combined_audio)
#                         logger.info(f"Created minimal audio file from first segment")
#                     except Exception as e:
#                         logger.error(f"Failed to create minimal audio: {e}")
#                         return None
#                 else:
#                     return None
        
#         # Check if combined audio was created
#         if not os.path.exists(combined_audio) or os.path.getsize(combined_audio) == 0:
#             logger.error("Failed to create combined audio file")
#             return None
            
#         logger.info(f"Combined audio created: {combined_audio}")
        
#         # Output path for dubbed video
#         video_filename = os.path.basename(video_path)
#         base_name, ext = os.path.splitext(video_filename)
#         dubbed_video = os.path.join(dubbing_dir, f"{base_name}_english_dubbed{ext}")
        
#         # Use ffmpeg to replace the audio track in the video
#         dub_cmd = [
#             "ffmpeg", "-y", 
#             "-i", video_path,
#             "-i", combined_audio,
#             "-map", "0:v",  # Use video from first input
#             "-map", "1:a",  # Use audio from second input
#             "-c:v", "copy", # Copy video stream without re-encoding
#             "-c:a", "aac",  # Convert audio to AAC
#             "-b:a", "192k", # Audio bitrate
#             "-shortest",    # End when the shortest input stream ends
#             dubbed_video
#         ]
        
#         # Run the dubbing command
#         logger.info("Creating dubbed video...")
#         process = await asyncio.create_subprocess_exec(
#             *dub_cmd,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
        
#         stdout, stderr = await process.communicate()
        
#         if process.returncode != 0:
#             logger.error(f"Error creating dubbed video: {stderr.decode()}")
#             return None
        
#         logger.info(f"Dubbed video created successfully: {dubbed_video}")
#         return dubbed_video
    
#     except Exception as e:
#         logger.error(f"Error synchronizing dubbed audio: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None


# # Add this function to dubbing.py to detect speaker gender

# async def detect_speakers_gender(transcript_segments, detected_language):
#     """
#     Analyze transcript to detect speaker gender based on language patterns.
    
#     Args:
#         transcript_segments: List of tuples (start_time, end_time, text)
#         detected_language: The detected language code
        
#     Returns:
#         Dictionary mapping segment indices to predicted gender ('male' or 'female')
#     """
#     import re
#     logger.info("Detecting speaker genders in transcript...")
    
#     # Initialize with a neutral stance
#     gender_predictions = {}
    
#     # Check for obvious speaker indicators first
#     speaker_pattern = re.compile(r'^\s*\[?([^:]+)(?:\])?:\s*(.*)', re.IGNORECASE)
    
#     # Common male and female name indicators (add more based on your common languages)
#     male_indicators = ['male', 'man', 'gentleman', 'boy', 'sir', 'mr', 'he', 'his', 'him',
#                       'father', 'brother', 'uncle', 'king', 'prince', 'narrator']
#     female_indicators = ['female', 'woman', 'lady', 'girl', 'madam', 'ms', 'mrs', 'miss', 'she', 'her', 'hers',
#                         'mother', 'sister', 'aunt', 'queen', 'princess', 'hostess']
    
#     # Language-specific gender patterns
#     language_patterns = {
#         'ar': {
#             'male': [r'\bهو\b', r'\bله\b', r'\bرجل\b'],  # Arabic male indicators
#             'female': [r'\bهي\b', r'\bلها\b', r'\bامرأة\b']  # Arabic female indicators
#         },
#         'es': {
#             'male': [r'\bél\b', r'\bsuyo\b', r'\bhombre\b'],  # Spanish male indicators
#             'female': [r'\bella\b', r'\bsuya\b', r'\bmujer\b']  # Spanish female indicators
#         },
#         'fr': {
#             'male': [r'\bil\b', r'\bson\b', r'\bhomme\b'],  # French male indicators
#             'female': [r'\belle\b', r'\bsa\b', r'\bfemme\b']  # French female indicators
#         }
#         # Add more languages as needed
#     }
    
#     # Get language-specific patterns if available
#     lang_male_patterns = language_patterns.get(detected_language, {}).get('male', [])
#     lang_female_patterns = language_patterns.get(detected_language, {}).get('female', [])
    
#     # Combine all patterns
#     all_male_patterns = lang_male_patterns + [rf'\b{indicator}\b' for indicator in male_indicators]
#     all_female_patterns = lang_female_patterns + [rf'\b{indicator}\b' for indicator in female_indicators]
    
#     # Track speaker continuity
#     speakers = {}  # Map speaker names to genders
#     current_speaker = None
    
#     # First pass: look for explicit speaker labels and gender indicators
#     for i, (_, _, text) in enumerate(transcript_segments):
#         # Look for speaker labels like "John: Hello" or "[John]: Hello"
#         match = speaker_pattern.match(text)
#         if match:
#             speaker_name = match.group(1).strip().lower()
#             if speaker_name not in speakers:
#                 # Check if the speaker name contains gender indicators
#                 speaker_gender = None
#                 if any(re.search(rf'\b{name}\b', speaker_name, re.IGNORECASE) for name in male_indicators):
#                     speaker_gender = 'male'
#                 elif any(re.search(rf'\b{name}\b', speaker_name, re.IGNORECASE) for name in female_indicators):
#                     speaker_gender = 'female'
                
#                 if speaker_gender:
#                     speakers[speaker_name] = speaker_gender
            
#             # Use known speaker gender if available
#             if speaker_name in speakers:
#                 gender_predictions[i] = speakers[speaker_name]
#                 current_speaker = speaker_name
#             continue
        
#         # If no speaker label, check for gender indicators in the text
#         male_score = sum(1 for pattern in all_male_patterns if re.search(pattern, text, re.IGNORECASE))
#         female_score = sum(1 for pattern in all_female_patterns if re.search(pattern, text, re.IGNORECASE))
        
#         if male_score > female_score:
#             gender_predictions[i] = 'male'
#         elif female_score > male_score:
#             gender_predictions[i] = 'female'
#         elif current_speaker in speakers:
#             # If no clear indicators, use previous speaker's gender
#             gender_predictions[i] = speakers[current_speaker]
    
#     # Second pass: Use clustering for segments without clear gender
#     # Fill in gaps based on surrounding segments
#     last_gender = None
#     for i in range(len(transcript_segments)):
#         if i not in gender_predictions:
#             # Look at surrounding segments
#             surrounding_male = 0
#             surrounding_female = 0
            
#             # Check previous 3 segments
#             for j in range(max(0, i-3), i):
#                 if j in gender_predictions:
#                     if gender_predictions[j] == 'male':
#                         surrounding_male += 1
#                     else:
#                         surrounding_female += 1
            
#             # Check next 3 segments
#             for j in range(i+1, min(len(transcript_segments), i+4)):
#                 if j in gender_predictions:
#                     if gender_predictions[j] == 'male':
#                         surrounding_male += 1
#                     else:
#                         surrounding_female += 1
            
#             # Assign gender based on surroundings
#             if surrounding_male > surrounding_female:
#                 gender_predictions[i] = 'male'
#             elif surrounding_female > surrounding_male:
#                 gender_predictions[i] = 'female'
#             elif last_gender:
#                 gender_predictions[i] = last_gender
#             else:
#                 # Default if no other information
#                 gender_predictions[i] = 'male'
        
#         last_gender = gender_predictions[i]
    
#     # Count males and females
#     male_count = sum(1 for gender in gender_predictions.values() if gender == 'male')
#     female_count = sum(1 for gender in gender_predictions.values() if gender == 'female')
    
#     logger.info(f"Gender detection complete. Found approximately {male_count} male and {female_count} female segments.")
    
#     return gender_predictions

# # Update the main dubbing function in dubbing.py

# async def create_english_dub(video_path, transcript_segments, detected_language, output_dir):
#     """
#     Main function to create English-dubbed version of a non-English video.
    
#     Args:
#         video_path: Path to the video file
#         transcript_segments: List of tuples (start_time, end_time, text)
#         detected_language: The detected language code
#         output_dir: Directory to save the output
        
#     Returns:
#         Path to the dubbed video
#     """
#     try:
#         # Skip if already in English
#         if detected_language == "en":
#             logger.info("Video is already in English, skipping dubbing")
#             return None, "Video is already in English, no dubbing needed."
        
#         start_time = time.time()
#         logger.info(f"Starting English dubbing process for {detected_language} video...")
        
#         # Step 1: Translate transcript to English
#         logger.info("Step 1/4: Translating transcript to English...")
#         translated_segments = await translate_transcript_to_english(transcript_segments, detected_language)
        
#         if not translated_segments:
#             return None, "Failed to translate transcript to English."
        
#         # Step 2: Detect speaker genders
#         logger.info("Step 2/4: Detecting speaker genders...")
#         gender_predictions = await detect_speakers_gender(transcript_segments, detected_language)
        
#         # Log gender distribution
#         male_count = sum(1 for gender in gender_predictions.values() if gender == 'male')
#         female_count = sum(1 for gender in gender_predictions.values() if gender == 'female')
#         logger.info(f"Detected {male_count} male segments and {female_count} female segments")
        
#         # Step 3: Generate English audio for each segment
#         logger.info("Step 3/4: Generating English audio with gender-appropriate voices...")
#         # Get video duration from last segment end time or default to 10 minutes
#         video_duration = translated_segments[-1][1] if translated_segments else 600
#         audio_dir = await generate_dubbed_audio(translated_segments, output_dir, video_duration, gender_predictions)
        
#         if not audio_dir:
#             return None, "Failed to generate English audio for dubbing."
        
#         # Step 4: Synchronize audio with video
#         logger.info("Step 4/4: Synchronizing audio with video...")
#         dubbed_video = await synchronize_dubbing(video_path, audio_dir, output_dir)
        
#         if not dubbed_video:
#             # Create a more specific error message
#             # Check if specific files exist to give better error messages
#             manifest_path = os.path.join(audio_dir, "manifest.json")
#             concat_file = os.path.join(audio_dir, "concat.txt")
            
#             if not os.path.exists(manifest_path):
#                 return None, "Failed to create audio manifest file."
#             elif not os.path.exists(concat_file):
#                 return None, "Failed to create audio concatenation file."
#             elif not os.path.exists(video_path):
#                 return None, f"Original video file not found: {video_path}"
#             else:
#                 return None, "Failed to synchronize audio with video."
        
#         end_time = time.time()
#         logger.info(f"English dubbing completed in {end_time - start_time:.2f} seconds")
        
#         # Check if the dubbed video was actually created
#         if not os.path.exists(dubbed_video):
#             return None, "Dubbed video file was not created."
            
#         # Check if the file size is reasonable (at least 1MB)
#         if os.path.getsize(dubbed_video) < 1024 * 1024:
#             return None, "Dubbed video file was created but appears to be incomplete."
        
#         # Return stats about the dubbing process
#         stats = {
#             "original_language": detected_language,
#             "segments_translated": len(translated_segments),
#             "male_segments": male_count,
#             "female_segments": female_count,
#             "duration": video_duration,
#             "processing_time": f"{end_time - start_time:.2f} seconds",
#             "file_size": f"{os.path.getsize(dubbed_video) / (1024 * 1024):.2f} MB"
#         }
        
#         return dubbed_video, stats
    
#     except Exception as e:
#         logger.error(f"Error in English dubbing process: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None, f"Error creating English dub: {str(e)}"

# import json  # Add missing import

from asyncio import subprocess
import os
# Change this line at the top of dubbing.py
import subprocess  # Instead of: from asyncio import subprocess
import time
import re
from logger_config import logger
from ffmpeg_check import verify_dependencies
# from gtts_audio_generator import generate_edge_tts
from utils import format_timestamp
import ollama

async def translate_transcript_to_english(transcript_segments, detected_language):
    """
    Translate the transcript segments to English using Deep Translator.
    
    Args:
        transcript_segments: List of tuples (start_time, end_time, text)
        detected_language: The detected language code
        
    Returns:
        List of tuples (start_time, end_time, translated_text)
    """
    if detected_language == "en":
        logger.info("Transcript already in English, no translation needed")
        return transcript_segments
    
    try:
        logger.info(f"Translating transcript from {detected_language} to English using Deep Translator...")
        
        # Import the translation library
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=detected_language, target='en')
        except ImportError:
            logger.warning("deep_translator not installed. Install with: pip install deep_translator")
            logger.warning("Falling back to Ollama for translation")
            return await translate_transcript_with_ollama(transcript_segments, detected_language)
        
        translated_segments = []
        batch_size = 25  # Deep translator recommends smaller batches
        
        # Process in batches for speed and to avoid rate limits
        for i in range(0, len(transcript_segments), batch_size):
            batch = transcript_segments[i:i+batch_size]
            
            # Process each segment in the batch
            for j, (start, end, text) in enumerate(batch):
                try:
                    # Skip empty or very short segments
                    if not text or len(text.strip()) < 2:
                        translated_segments.append((start, end, text))
                        continue
                    
                    # Translate the segment
                    translated_text = translator.translate(text)
                    translated_segments.append((start, end, translated_text))
                    
                    # Log progress periodically
                    if (i + j) % 50 == 0:
                        logger.info(f"Translated {i + j}/{len(transcript_segments)} segments")
                except Exception as e:
                    logger.error(f"Error translating segment {i+j}: {e}")
                    # Keep original text as fallback
                    translated_segments.append((start, end, text))
            
           
        
        logger.info(f"Translation completed. Translated {len(translated_segments)} segments.")
        return translated_segments
    
    except Exception as e:
        logger.error(f"Error in Deep Translator: {e}")
        # Fall back to Ollama translation if Google fails
        logger.info("Falling back to Ollama for translation")
        return await translate_transcript_with_ollama(transcript_segments, detected_language)

async def translate_transcript_with_ollama(transcript_segments, detected_language):
    """
    Fallback translation using Ollama LLM.
    """
    try:
        logger.info(f"Translating transcript from {detected_language} to English using Ollama...")
        
        # Prepare chunks of transcript for translation (to avoid token limits)
        chunk_size = 20  # Number of segments per chunk
        chunks = [transcript_segments[i:i + chunk_size] for i in range(0, len(transcript_segments), chunk_size)]
        
        translated_segments = []
        
        for i, chunk in enumerate(chunks):
            # Extract text from segments
            text_list = [seg[2] for seg in chunk]
            text_with_indices = "\n".join([f"[{idx}] {text}" for idx, text in enumerate(text_list)])
            
            # Create translation prompt
            prompt = f"""
            Translate the following text segments from {detected_language} to English.
            Maintain the original meaning and tone. Keep all numbers, names, and technical terms intact.
            The translations should be natural and fluent English, not literal translations.
            
            Return only the translations with their index numbers in this exact format:
            [0] English translation of first segment
            [1] English translation of second segment
            ...and so on.
            
            Text to translate:
            {text_with_indices}
            """
            
            # Call Ollama for translation
            logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} segments)")
            response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "system", "content": prompt}])
            result = response["message"]["content"]
            
            # Parse the translations
            translations = {}
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                match = re.match(r'\[(\d+)\](.*)', line)
                if match:
                    index = int(match.group(1))
                    translated_text = match.group(2).strip()
                    translations[index] = translated_text
            
            # Map translations back to original segments with timing intact
            for j, (start, end, _) in enumerate(chunk):
                translated_text = translations.get(j, "")
                if not translated_text:
                    # Fallback if translation is missing
                    translated_text = chunk[j][2]
                    logger.warning(f"Missing translation for segment {j} in chunk {i}, using original")
                
                translated_segments.append((start, end, translated_text))
        
        logger.info(f"Translation completed. Translated {len(translated_segments)} segments.")
        return translated_segments
    
    except Exception as e:
        logger.error(f"Error translating with Ollama: {e}")
        # Return original if translation fails
        return transcript_segments

# Update the generate_dubbed_audio function in dubbing.py

# Update the generate_dubbed_audio function to use only one voice per gender

async def generate_dubbed_audio(translated_segments, output_dir, video_duration, gender_predictions=None):
    """
    Generate English audio for each translated segment with one consistent voice per gender.
    
    Args:
        translated_segments: List of tuples (start_time, end_time, translated_text)
        output_dir: Directory to save audio files
        video_duration: Duration of the video in seconds
        gender_predictions: Dictionary mapping segment indices to predicted gender
        
    Returns:
        Path to the directory containing audio segments
    """
    try:
        deps_ok, deps_message = verify_dependencies()
        if not deps_ok:
            logger.error(f"Dependency check failed: {deps_message}")
            return None
            
        # Create temp directory for audio segments
        temp_dir = os.path.join(output_dir, "dubbing", "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set up one voice per gender - use the most reliable voices
        male_voice = "en-US-GuyNeural"    # Single consistent male voice
        female_voice = "en-US-JennyNeural" # Single consistent female voice
        
        # Default to alternating genders if no gender predictions
        if not gender_predictions:
            logger.info("No gender predictions provided. Using consistent male/female voices based on segment index.")
            gender_predictions = {i: 'male' if i % 2 == 0 else 'female' for i in range(len(translated_segments))}
        
        logger.info(f"Generating English audio for {len(translated_segments)} segments with consistent gender voices...")
        logger.info(f"Using '{male_voice}' for all male segments and '{female_voice}' for all female segments")
        
        # Track the progress
        total_segments = len(translated_segments)
        audio_segment_paths = []
        
        # Try using Edge TTS if available, with fallback to gTTS
        try:
            import edge_tts
            edge_tts_available = True
            logger.info("Using Edge TTS for gender-appropriate voices")
        except ImportError:
            edge_tts_available = False
            logger.info("Edge TTS not available - falling back to gTTS (without gender distinction)")
            from gtts import gTTS
        
        # Generate audio for each segment
        for i, (start, end, text) in enumerate(translated_segments):
            segment_duration = end - start
            segment_path = os.path.join(temp_dir, f"segment_{i:04d}_{start:.2f}_{end:.2f}.mp3")
            
            # Skip empty segments
            if not text.strip():
                logger.info(f"Skipping empty segment {i}/{total_segments}")
                continue
            
            try:
                if i % 10 == 0:  # Log progress every 10 segments
                    logger.info(f"Generating audio segment {i+1}/{total_segments}")
                
                # Determine gender for this segment
                gender = gender_predictions.get(i, 'male')  # Default to male if not specified
                
                # Select voice based on gender - only one voice per gender
                selected_voice = male_voice if gender == 'male' else female_voice
                
                if edge_tts_available:
                    try:
                        # Generate with Edge TTS
                        logger.info(f"Using {gender} voice: {selected_voice}")
                        communicate = edge_tts.Communicate(text, selected_voice)
                        await communicate.save(segment_path)
                    except Exception as e:
                        logger.error(f"Edge TTS failed: {e}. Falling back to gTTS")
                        # Fallback to gTTS
                        tts = gTTS(text=text, lang='en', slow=False)
                        tts.save(segment_path)
                else:
                    # Use gTTS (no gender distinction)
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(segment_path)
                
                # Store the path along with timing information
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    audio_segment_paths.append({
                        "path": segment_path,
                        "start": start,
                        "end": end,
                        "text": text,
                        "gender": gender
                    })
                else:
                    logger.warning(f"Segment {i} audio file is empty or not created")
            except Exception as e:
                logger.error(f"Error generating audio for segment {i}: {e}")
                # Try with a shorter text if original fails
                try:
                    shortened_text = text[:200] + "..." if len(text) > 200 else text
                    tts = gTTS(text=shortened_text, lang='en', slow=False)
                    tts.save(segment_path)
                    
                    if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                        audio_segment_paths.append({
                            "path": segment_path,
                            "start": start,
                            "end": end,
                            "text": shortened_text,
                            "gender": gender
                        })
                except Exception as inner_e:
                    logger.error(f"Error with shortened text for segment {i}: {inner_e}")
        
        logger.info(f"Generated {len(audio_segment_paths)} audio segments")
        
        # Create a manifest file with timing information
        manifest_path = os.path.join(temp_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                "segments": audio_segment_paths,
                "video_duration": video_duration,
                "voice_settings": {
                    "male_voice": male_voice,
                    "female_voice": female_voice
                }
            }, f, indent=2)
        
        return temp_dir
    
    except Exception as e:
        logger.error(f"Error generating dubbed audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Update the synchronize_dubbing function in dubbing.py

def synchronize_dubbing(video_path, audio_segments_dir, output_dir):
    """
    Synchronize the dubbed audio with the original video.
    
    Args:
        video_path: Path to the original video
        audio_segments_dir: Directory containing audio segments
        output_dir: Directory to save the dubbed video
        
    Returns:
        Path to the dubbed video
    """
    try:
        # Load the audio segment manifest
        manifest_path = os.path.join(audio_segments_dir, "manifest.json")
        
        # Check if manifest exists
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found at {manifest_path}")
            return None
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        segments = manifest["segments"]
        video_duration = manifest["video_duration"]
        
        # Create output directory
        dubbing_dir = os.path.join(output_dir, "dubbing")
        os.makedirs(dubbing_dir, exist_ok=True)
        
        # Make sure we have some segments to work with
        if not segments:
            logger.error("No audio segments found in manifest")
            return None
            
        logger.info(f"Found {len(segments)} audio segments to process")
        
        # First, let's verify all segment files exist
        existing_segments = []
        for segment in segments:
            path = segment["path"]
            if os.path.exists(path) and os.path.getsize(path) > 0:
                existing_segments.append(segment)
            else:
                logger.warning(f"Segment file not found or empty: {path}")
        
        if not existing_segments:
            logger.error("No valid audio segments found")
            return None
            
        logger.info(f"{len(existing_segments)} valid audio segments found")
        
        # Create a file list for concatenation with absolute paths
        concat_file = os.path.join(audio_segments_dir, "concat.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for segment in existing_segments:
                # Use absolute path to avoid directory issues
                abs_path = os.path.abspath(segment["path"])
                # Escape backslashes in Windows paths
                safe_path = abs_path.replace('\\', '\\\\')
                f.write(f"file '{safe_path}'\n")
        
        # Output path for combined audio
        combined_audio = os.path.join(audio_segments_dir, "combined_audio.mp3")
        
        # Use ffmpeg to concatenate all audio files
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", combined_audio
        ]
        
        # Run the concatenation command
        logger.info("Combining audio segments...")
        logger.info(f"Using concat file: {concat_file}")
        
        # Check if concat file exists and has content
        if os.path.exists(concat_file):
            with open(concat_file, 'r') as f:
                logger.info(f"Concat file first few lines: {f.readline()[:100]}")
        else:
            logger.error(f"Concat file does not exist: {concat_file}")
            return None
        
        # Replace asyncio.create_subprocess_exec with subprocess.run
        process_result = subprocess.run(
            concat_cmd,
            capture_output=True,
            text=True
        )
        
        if process_result.returncode != 0:
            logger.error(f"Error concatenating audio: {process_result.stderr}")
            
            # Try alternative approach using a different concat method
            logger.info("Trying alternative audio concatenation method...")
            
            # Use direct file inputs instead of a concat file
            alt_concat_cmd = ["ffmpeg", "-y"]
            
            # Add each file as an input
            for segment in existing_segments[:10]:  # Limit to first 10 for testing
                alt_concat_cmd.extend(["-i", segment["path"]])
            
            # Add filter complex to merge all inputs
            filter_str = ""
            for i in range(min(10, len(existing_segments))):
                filter_str += f"[{i}:0]"
            filter_str += f"concat=n={min(10, len(existing_segments))}:v=0:a=1[out]"
            
            alt_concat_cmd.extend([
                "-filter_complex", filter_str,
                "-map", "[out]",
                combined_audio
            ])
            
            # Run the alternative command
            process_result = subprocess.run(
                alt_concat_cmd,
                capture_output=True,
                text=True
            )
            
            if process_result.returncode != 0:
                logger.error(f"Alternative audio concatenation failed: {process_result.stderr}")
                
                # If both methods fail, try creating a minimal audio file for testing
                logger.info("Creating minimal audio for testing...")
                if len(existing_segments) > 0:
                    import shutil
                    try:
                        # Just copy the first segment as the combined audio
                        shutil.copy(existing_segments[0]["path"], combined_audio)
                        logger.info(f"Created minimal audio file from first segment")
                    except Exception as e:
                        logger.error(f"Failed to create minimal audio: {e}")
                        return None
                else:
                    return None
        
        # Check if combined audio was created
        if not os.path.exists(combined_audio) or os.path.getsize(combined_audio) == 0:
            logger.error("Failed to create combined audio file")
            return None
            
        logger.info(f"Combined audio created: {combined_audio}")
        
        # Output path for dubbed video
        video_filename = os.path.basename(video_path)
        base_name, ext = os.path.splitext(video_filename)
        dubbed_video = os.path.join(dubbing_dir, f"{base_name}_english_dubbed{ext}")
        
        # Use ffmpeg to replace the audio track in the video
        dub_cmd = [
            "ffmpeg", "-y", 
            "-i", video_path,
            "-i", combined_audio,
            "-map", "0:v",  # Use video from first input
            "-map", "1:a",  # Use audio from second input
            "-c:v", "copy", # Copy video stream without re-encoding
            "-c:a", "aac",  # Convert audio to AAC
            "-b:a", "192k", # Audio bitrate
            "-shortest",    # End when the shortest input stream ends
            dubbed_video
        ]
        
        # Run the dubbing command
        logger.info("Creating dubbed video...")
        
        # Replace asyncio.create_subprocess_exec with subprocess.run
        process_result = subprocess.run(
            dub_cmd,
            capture_output=True,
            text=True
        )
        
        if process_result.returncode != 0:
            logger.error(f"Error creating dubbed video: {process_result.stderr}")
            return None
        
        logger.info(f"Dubbed video created successfully: {dubbed_video}")
        return dubbed_video
    
    except Exception as e:
        logger.error(f"Error synchronizing dubbed audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# Add this function to dubbing.py to detect speaker gender

async def detect_speakers_gender(transcript_segments, detected_language):
    """
    Analyze transcript to detect speaker gender based on language patterns.
    
    Args:
        transcript_segments: List of tuples (start_time, end_time, text)
        detected_language: The detected language code
        
    Returns:
        Dictionary mapping segment indices to predicted gender ('male' or 'female')
    """
    import re
    logger.info("Detecting speaker genders in transcript...")
    
    # Initialize with a neutral stance
    gender_predictions = {}
    
    # Check for obvious speaker indicators first
    speaker_pattern = re.compile(r'^\s*\[?([^:]+)(?:\])?:\s*(.*)', re.IGNORECASE)
    
    # Common male and female name indicators (add more based on your common languages)
    male_indicators = ['male', 'man', 'gentleman', 'boy', 'sir', 'mr', 'he', 'his', 'him',
                      'father', 'brother', 'uncle', 'king', 'prince', 'narrator']
    female_indicators = ['female', 'woman', 'lady', 'girl', 'madam', 'ms', 'mrs', 'miss', 'she', 'her', 'hers',
                        'mother', 'sister', 'aunt', 'queen', 'princess', 'hostess']
    
    # Language-specific gender patterns
    language_patterns = {
        'ar': {
            'male': [r'\bهو\b', r'\bله\b', r'\bرجل\b'],  # Arabic male indicators
            'female': [r'\bهي\b', r'\bلها\b', r'\bامرأة\b']  # Arabic female indicators
        },
        'es': {
            'male': [r'\bél\b', r'\bsuyo\b', r'\bhombre\b'],  # Spanish male indicators
            'female': [r'\bella\b', r'\bsuya\b', r'\bmujer\b']  # Spanish female indicators
        },
        'fr': {
            'male': [r'\bil\b', r'\bson\b', r'\bhomme\b'],  # French male indicators
            'female': [r'\belle\b', r'\bsa\b', r'\bfemme\b']  # French female indicators
        }
        # Add more languages as needed
    }
    
    # Get language-specific patterns if available
    lang_male_patterns = language_patterns.get(detected_language, {}).get('male', [])
    lang_female_patterns = language_patterns.get(detected_language, {}).get('female', [])
    
    # Combine all patterns
    all_male_patterns = lang_male_patterns + [rf'\b{indicator}\b' for indicator in male_indicators]
    all_female_patterns = lang_female_patterns + [rf'\b{indicator}\b' for indicator in female_indicators]
    
    # Track speaker continuity
    speakers = {}  # Map speaker names to genders
    current_speaker = None
    
    # First pass: look for explicit speaker labels and gender indicators
    for i, (_, _, text) in enumerate(transcript_segments):
        # Look for speaker labels like "John: Hello" or "[John]: Hello"
        match = speaker_pattern.match(text)
        if match:
            speaker_name = match.group(1).strip().lower()
            if speaker_name not in speakers:
                # Check if the speaker name contains gender indicators
                speaker_gender = None
                if any(re.search(rf'\b{name}\b', speaker_name, re.IGNORECASE) for name in male_indicators):
                    speaker_gender = 'male'
                elif any(re.search(rf'\b{name}\b', speaker_name, re.IGNORECASE) for name in female_indicators):
                    speaker_gender = 'female'
                
                if speaker_gender:
                    speakers[speaker_name] = speaker_gender
            
            # Use known speaker gender if available
            if speaker_name in speakers:
                gender_predictions[i] = speakers[speaker_name]
                current_speaker = speaker_name
            continue
        
        # If no speaker label, check for gender indicators in the text
        male_score = sum(1 for pattern in all_male_patterns if re.search(pattern, text, re.IGNORECASE))
        female_score = sum(1 for pattern in all_female_patterns if re.search(pattern, text, re.IGNORECASE))
        
        if male_score > female_score:
            gender_predictions[i] = 'male'
        elif female_score > male_score:
            gender_predictions[i] = 'female'
        elif current_speaker in speakers:
            # If no clear indicators, use previous speaker's gender
            gender_predictions[i] = speakers[current_speaker]
    
    # Second pass: Use clustering for segments without clear gender
    # Fill in gaps based on surrounding segments
    last_gender = None
    for i in range(len(transcript_segments)):
        if i not in gender_predictions:
            # Look at surrounding segments
            surrounding_male = 0
            surrounding_female = 0
            
            # Check previous 3 segments
            for j in range(max(0, i-3), i):
                if j in gender_predictions:
                    if gender_predictions[j] == 'male':
                        surrounding_male += 1
                    else:
                        surrounding_female += 1
            
            # Check next 3 segments
            for j in range(i+1, min(len(transcript_segments), i+4)):
                if j in gender_predictions:
                    if gender_predictions[j] == 'male':
                        surrounding_male += 1
                    else:
                        surrounding_female += 1
            
            # Assign gender based on surroundings
            if surrounding_male > surrounding_female:
                gender_predictions[i] = 'male'
            elif surrounding_female > surrounding_male:
                gender_predictions[i] = 'female'
            elif last_gender:
                gender_predictions[i] = last_gender
            else:
                # Default if no other information
                gender_predictions[i] = 'male'
        
        last_gender = gender_predictions[i]
    
    # Count males and females
    male_count = sum(1 for gender in gender_predictions.values() if gender == 'male')
    female_count = sum(1 for gender in gender_predictions.values() if gender == 'female')
    
    logger.info(f"Gender detection complete. Found approximately {male_count} male and {female_count} female segments.")
    
    return gender_predictions

# Update the main dubbing function in dubbing.py

async def create_english_dub(video_path, transcript_segments, detected_language, output_dir):
    """
    Main function to create English-dubbed version of a non-English video.
    
    Args:
        video_path: Path to the video file
        transcript_segments: List of tuples (start_time, end_time, text)
        detected_language: The detected language code
        output_dir: Directory to save the output
        
    Returns:
        Path to the dubbed video
    """
    try:
        # Skip if already in English
        if detected_language == "en":
            logger.info("Video is already in English, skipping dubbing")
            return None, "Video is already in English, no dubbing needed."
        
        start_time = time.time()
        logger.info(f"Starting English dubbing process for {detected_language} video...")
        
        # Step 1: Translate transcript to English
        logger.info("Step 1/4: Translating transcript to English...")
        translated_segments = await translate_transcript_to_english(transcript_segments, detected_language)
        
        if not translated_segments:
            return None, "Failed to translate transcript to English."
        
        # Step 2: Detect speaker genders
        logger.info("Step 2/4: Detecting speaker genders...")
        gender_predictions = await detect_speakers_gender(transcript_segments, detected_language)
        
        # Log gender distribution
        male_count = sum(1 for gender in gender_predictions.values() if gender == 'male')
        female_count = sum(1 for gender in gender_predictions.values() if gender == 'female')
        logger.info(f"Detected {male_count} male segments and {female_count} female segments")
        
        # Step 3: Generate English audio for each segment
        logger.info("Step 3/4: Generating English audio with gender-appropriate voices...")
        # Get video duration from last segment end time or default to 10 minutes
        video_duration = translated_segments[-1][1] if translated_segments else 600
        audio_dir = await generate_dubbed_audio(translated_segments, output_dir, video_duration, gender_predictions)
        
        if not audio_dir:
            return None, "Failed to generate English audio for dubbing."
        
        # Step 4: Synchronize audio with video
        logger.info("Step 4/4: Synchronizing audio with video...")
        dubbed_video = synchronize_dubbing(video_path, audio_dir, output_dir)
        
        if not dubbed_video:
            # Create a more specific error message
            # Check if specific files exist to give better error messages
            manifest_path = os.path.join(audio_dir, "manifest.json")
            concat_file = os.path.join(audio_dir, "concat.txt")
            
            if not os.path.exists(manifest_path):
                return None, "Failed to create audio manifest file."
            elif not os.path.exists(concat_file):
                return None, "Failed to create audio concatenation file."
            elif not os.path.exists(video_path):
                return None, f"Original video file not found: {video_path}"
            else:
                return None, "Failed to synchronize audio with video."
        
        end_time = time.time()
        logger.info(f"English dubbing completed in {end_time - start_time:.2f} seconds")
        
        # Check if the dubbed video was actually created
        if not os.path.exists(dubbed_video):
            return None, "Dubbed video file was not created."
            
        # Check if the file size is reasonable (at least 1MB)
        if os.path.getsize(dubbed_video) < 1024 * 1024:
            return None, "Dubbed video file was created but appears to be incomplete."
        
        # Return stats about the dubbing process
        stats = {
            "original_language": detected_language,
            "segments_translated": len(translated_segments),
            "male_segments": male_count,
            "female_segments": female_count,
            "duration": video_duration,
            "processing_time": f"{end_time - start_time:.2f} seconds",
            "file_size": f"{os.path.getsize(dubbed_video) / (1024 * 1024):.2f} MB"
        }
        
        return dubbed_video, stats
    
    except Exception as e:
        logger.error(f"Error in English dubbing process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error creating English dub: {str(e)}"

import json  # Add missing import