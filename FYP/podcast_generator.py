import re
import time
import ollama
import asyncio
import os
import json
import hashlib
import functools
from constants import OUTPUT_DIR
from logger_config import logger
from utils import format_time_duration
import google.generativeai as genai
from dotenv import load_dotenv


# Cache for podcast responses
_podcast_script_cache = {}
_cache_size_limit = 20  # Limit cache size to prevent memory issues

# async def generate_podcast_script(transcript_segments, video_info, detected_language="en", custom_prompt=None):
#     """
#     Generate a conversational podcast script between two hosts based on video content.
#     """
#     start_time = time.time()
    
#     # Create a cache key from video info
#     video_title = video_info.get('title', 'Unknown')
#     cache_key = hashlib.md5(f"{video_title}:{detected_language}:{custom_prompt or ''}".encode()).hexdigest()
    
#     # Check if response is in cache
#     if cache_key in _podcast_script_cache:
#         logger.info("Using cached podcast script (saved LLM call)")
#         return _podcast_script_cache[cache_key]
    
#     # Prepare transcript for analysis
#     if not transcript_segments:
#         return {"error": "No transcript provided for podcast generation"}
    
#     # Extract full text from transcript segments
#     full_text = " ".join([seg[2] for seg in transcript_segments])
    
#     # Truncate very long transcripts to avoid token limits
#     max_length = 12000  # Characters
#     if len(full_text) > max_length:
#         logger.info(f"Transcript too long ({len(full_text)} chars), truncating to {max_length} chars")
#         # Keep beginning, middle and end for better context
#         third = max_length // 3
#         full_text = full_text[:third] + "..." + full_text[len(full_text)//2-third//2:len(full_text)//2+third//2] + "..." + full_text[-third:]
    
#     # Determine podcast host names based on language
#     if detected_language == "en":
#         host1_name = "Alex"
#         host2_name = "Jamie"
#     else:
#         # Use more international names for non-English podcasts
#         host1_name = "Ana"
#         host2_name = "Kai"
    
#     # Build host personalities based on video content
#     host1_personality = "curious and analytical"
#     host2_personality = "knowledgeable and enthusiastic"
    
#     # Get video duration estimate (rough approximation from transcript)
#     video_duration = transcript_segments[-1][1] if transcript_segments else 600  # Default to 10 minutes
    
#     # Customize podcast style based on video content
#     podcast_style = "informative and engaging"
#     if custom_prompt:
#         # User provided custom instructions
#         user_preferences = custom_prompt
#     else:
#         # Automatic customization
#         title = video_info.get('title', '').lower()
#         desc = video_info.get('description', '').lower()
        
#         if any(term in title or term in desc for term in ['tutorial', 'how to', 'learn', 'guide']):
#             podcast_style = "educational and practical"
#             host1_personality = "curious beginner with thoughtful questions"
#             host2_personality = "patient expert with clear explanations"
#         elif any(term in title or term in desc for term in ['review', 'analysis', 'critique']):
#             podcast_style = "analytical and opinion-focused"
#             host1_personality = "skeptical and detail-oriented"
#             host2_personality = "balanced and considering multiple perspectives"
#         elif any(term in title or term in desc for term in ['news', 'update', 'current events']):
#             podcast_style = "informative and timely"
#             host1_personality = "inquisitive interviewer"
#             host2_personality = "well-informed commentator"
        
#         user_preferences = f"Create a {podcast_style} conversation"
    
#     # Build the prompt
#     prompt = f"""
#     You are an expert podcast scriptwriter who specializes in creating engaging conversational podcasts.
#     Your task is to create a natural-sounding podcast script between two hosts ({host1_name} and {host2_name}) 
#     who are discussing the content from a video.

#     VIDEO DETAILS:
#     - Title: "{video_info.get('title', 'Unknown')}"
#     - Duration: {format_time_duration(video_duration)} ({video_duration:.1f}s)
#     - Description: "{(video_info.get('description') or '')[:300]}..."

#     PODCAST REQUIREMENTS:
#     1. Create a 7-10 minute podcast script (approximately 800-1000 words) in conversation format.
#     2. The hosts should have distinct personalities:
#        - {host1_name}: {host1_personality}
#        - {host2_name}: {host2_personality}
#     3. The conversation must feel natural, not like they're directly summarizing a video.
#     4. Include an introduction, 3-5 main discussion points, and a conclusion.
#     5. The hosts should engage with each other (asking questions, building on ideas, occasionally disagreeing).
#     6. {user_preferences}
#     7. Don't reference "the video" or "the speaker in the video" - the hosts should discuss the topic as their own conversation.
#     8. Format the script clearly with speaker names followed by their dialogue.
    
#     TRANSCRIPT CONTENT TO DISCUSS:
#     {full_text[:12000]}
    
#     RETURN FORMAT:
#     Please provide the podcast script in the following JSON format:
#     {{
#       "title": "Engaging podcast title",
#       "description": "Brief description of the podcast episode",
#       "hosts": ["{host1_name}", "{host2_name}"],
#       "estimated_duration_minutes": X,
#       "script": [
#         {{
#           "speaker": "{host1_name}", 
#           "text": "Welcome to the show..."
#         }},
#         {{
#           "speaker": "{host2_name}",
#           "text": "Great to be here..."
#         }},
#         ...
#       ]
#     }}
#     """
    
#     try:
#         logger.info("Generating podcast script with LLM...")
        
#         # Run Ollama in a thread pool to prevent blocking
#         loop = asyncio.get_event_loop()
#         response = await loop.run_in_executor(
#             None,
#             functools.partial(
#                 ollama.chat,
#                 model="deepseek-r1:latest",
#                 messages=[{"role": "system", "content": prompt}]
#             )
#         )
        
#         end_time = time.time()
#         logger.info(f"Time taken to generate podcast script: {end_time - start_time:.4f} seconds")
        
#         raw_content = response["message"]["content"]
        
#         # Extract JSON from response
#         podcast_data = None
        
#         # Try direct JSON parsing
#         try:
#             podcast_data = json.loads(raw_content)
#         except json.JSONDecodeError:
#             logger.info("Direct JSON parsing failed, trying regex extraction...")
            
#             # Try regex extraction
#             json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*\})', raw_content.replace('\n', ' '), re.DOTALL)
#             if json_match:
#                 try:
#                     json_str = json_match.group(1) or json_match.group(2)
#                     podcast_data = json.loads(json_str)
#                 except json.JSONDecodeError:
#                     logger.warning("Regex JSON extraction failed")
            
#             # If still no valid JSON, try to fix common issues
#             if not podcast_data:
#                 logger.info("Trying to fix and extract JSON...")
#                 # Remove markdown code blocks
#                 cleaned = re.sub(r'```json\s*|\s*```', '', raw_content)
#                 # Try to find object bounds
#                 start_idx = cleaned.find('{')
#                 end_idx = cleaned.rfind('}')
#                 if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
#                     try:
#                         json_str = cleaned[start_idx:end_idx+1]
#                         podcast_data = json.loads(json_str)
#                     except json.JSONDecodeError:
#                         logger.warning("Fixed JSON extraction failed")
        
#         # If JSON parsing fails, create a structured format manually
#         if not podcast_data or not isinstance(podcast_data, dict) or "script" not in podcast_data:
#             logger.warning("Failed to parse podcast data, creating structured format manually")
            
#             # Extract title if possible
#             title_match = re.search(r'"title":\s*"([^"]+)"', raw_content)
#             podcast_title = title_match.group(1) if title_match else f"Podcast about {video_info.get('title', 'Video Topic')}"
            
#             # Split by speaker indicators
#             script_lines = []
#             current_speaker = None
#             current_text = []
            
#             for line in raw_content.split('\n'):
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 speaker_match = re.match(r'^({}|{}):\s*(.+)'.format(host1_name, host2_name), line)
#                 if speaker_match:
#                     # Save previous speaker's text if exists
#                     if current_speaker and current_text:
#                         script_lines.append({
#                             "speaker": current_speaker,
#                             "text": " ".join(current_text)
#                         })
#                         current_text = []
                    
#                     # Start new speaker
#                     current_speaker = speaker_match.group(1)
#                     current_text.append(speaker_match.group(2))
#                 elif current_speaker:
#                     # Continue current speaker's text
#                     current_text.append(line)
            
#             # Add final speaker text
#             if current_speaker and current_text:
#                 script_lines.append({
#                     "speaker": current_speaker,
#                     "text": " ".join(current_text)
#                 })
            
#             # Create structured data
#             podcast_data = {
#                 "title": podcast_title,
#                 "description": f"A conversation about {video_info.get('title', 'this interesting topic')}",
#                 "hosts": [host1_name, host2_name],
#                 "estimated_duration_minutes": 5,
#                 "script": script_lines if script_lines else [
#                     {"speaker": host1_name, "text": f"Welcome to our discussion about {video_info.get('title', 'this topic')}!"},
#                     {"speaker": host2_name, "text": "I'm excited to dive into this with you today."}
#                 ]
#             }
        
#         # Cache the result
#         if len(_podcast_script_cache) >= _cache_size_limit:
#             # Remove an arbitrary entry if cache is full
#             oldest_key = next(iter(_podcast_script_cache))
#             del _podcast_script_cache[oldest_key]
        
#         _podcast_script_cache[cache_key] = podcast_data
        
#         return podcast_data
    
#     except Exception as e:
#         logger.error(f"Error generating podcast script: {e}")
#         return {
#             "error": f"Failed to generate podcast: {str(e)}",
#             "title": f"Podcast about {video_info.get('title', 'Video Topic')}",
#             "hosts": [host1_name, host2_name],
#             "script": [
#                 {"speaker": host1_name, "text": f"Welcome to our discussion about {video_info.get('title', 'this topic')}!"},
#                 {"speaker": host2_name, "text": "Unfortunately, we're experiencing some technical difficulties with our script generation."}
#             ]
#         }


async def generate_podcast_script(transcript_segments, video_info, detected_language="en", custom_prompt=None):
    """
    Generate a conversational podcast script between two hosts based on video content using Google Gemini API.
    """
    start_time = time.time()
    
    # Create a cache key from video info
    video_title = video_info.get('title', 'Unknown')
    cache_key = hashlib.md5(f"{video_title}:{detected_language}:{custom_prompt or ''}".encode()).hexdigest()
    
    # Check if response is in cache
    if cache_key in _podcast_script_cache:
        logger.info("Using cached podcast script (saved LLM call)")
        return _podcast_script_cache[cache_key]
    
    # Prepare transcript for analysis
    if not transcript_segments:
        return {"error": "No transcript provided for podcast generation"}
    
    # Extract full text from transcript segments
    full_text = " ".join([seg[2] for seg in transcript_segments])
    
    # Truncate very long transcripts to avoid token limits
    max_length = 12000  # Characters
    if len(full_text) > max_length:
        logger.info(f"Transcript too long ({len(full_text)} chars), truncating to {max_length} chars")
        # Keep beginning, middle and end for better context
        third = max_length // 3
        full_text = full_text[:third] + "..." + full_text[len(full_text)//2-third//2:len(full_text)//2+third//2] + "..." + full_text[-third:]
    
    # Determine podcast host names based on language
    if detected_language == "en":
        host1_name = "Alex"
        host2_name = "Jamie"
    else:
        # Use more international names for non-English podcasts
        host1_name = "Ana"
        host2_name = "Kai"
    
    # Build host personalities based on video content
    host1_personality = "curious and analytical"
    host2_personality = "knowledgeable and enthusiastic"
    
    # Get video duration estimate (rough approximation from transcript)
    video_duration = transcript_segments[-1][1] if transcript_segments else 600  # Default to 10 minutes
    
    # Customize podcast style based on video content
    podcast_style = "informative and engaging"
    if custom_prompt:
        # User provided custom instructions
        user_preferences = custom_prompt
    else:
        # Automatic customization
        title = video_info.get('title', '').lower()
        desc = video_info.get('description', '').lower()
        
        if any(term in title or term in desc for term in ['tutorial', 'how to', 'learn', 'guide']):
            podcast_style = "educational and practical"
            host1_personality = "curious beginner with thoughtful questions"
            host2_personality = "patient expert with clear explanations"
        elif any(term in title or term in desc for term in ['review', 'analysis', 'critique']):
            podcast_style = "analytical and opinion-focused"
            host1_personality = "skeptical and detail-oriented"
            host2_personality = "balanced and considering multiple perspectives"
        elif any(term in title or term in desc for term in ['news', 'update', 'current events']):
            podcast_style = "informative and timely"
            host1_personality = "inquisitive interviewer"
            host2_personality = "well-informed commentator"
        
        user_preferences = f"Create a {podcast_style} conversation"
    
    # # Build the prompt
    # prompt = f"""
    # You are an expert podcast scriptwriter who specializes in creating engaging conversational podcasts.
    # Your task is to create a natural-sounding podcast script between two hosts ({host1_name} and {host2_name}) 
    # who are discussing the content from a video.

    # VIDEO DETAILS:
    # - Title: "{video_info.get('title', 'Unknown')}"
    # - Duration: {format_time_duration(video_duration)} ({video_duration:.1f}s)
    # - Description: "{(video_info.get('description') or '')[:300]}..."

    # PODCAST REQUIREMENTS:
    # 1. Create a 7-10 minute podcast script (approximately 800-1000 words) in conversation format.
    # 2. The hosts should have distinct personalities:
    #    - {host1_name}: {host1_personality}
    #    - {host2_name}: {host2_personality}
    # 3. The conversation must feel natural, not like they're directly summarizing a video.
    # 4. Include an introduction, 3-5 main discussion points, and a conclusion.
    # 5. The hosts should engage with each other (asking questions, building on ideas, occasionally disagreeing).
    # 6. {user_preferences}
    # 7. Don't reference "the video" or "the speaker in the video" - the hosts should discuss the topic as their own conversation.
    # 8. Format the script clearly with speaker names followed by their dialogue.
    
    # TRANSCRIPT CONTENT TO DISCUSS:
    # {full_text[:12000]}
    
    # RETURN FORMAT:
    # Please provide the podcast script in the following JSON format:
    # {{
    #   "title": "Engaging podcast title",
    #   "description": "Brief description of the podcast episode",
    #   "hosts": ["{host1_name}", "{host2_name}"],
    #   "estimated_duration_minutes": X,
    #   "script": [
    #     {{
    #       "speaker": "{host1_name}", 
    #       "text": "Welcome to the show..."
    #     }},
    #     {{
    #       "speaker": "{host2_name}",
    #       "text": "Great to be here..."
    #     }},
    #     ...
    #   ]
    # }}
    # """

    # Update this in your generate_podcast_script function:

    prompt = f"""
    You are an expert podcast scriptwriter who specializes in creating engaging conversational podcasts.
    Your task is to create a natural-sounding podcast script between two hosts ({host1_name} and {host2_name}) 
    who are discussing the content from a video.

    VIDEO DETAILS:
    - Title: "{video_info.get('title', 'Unknown')}"
    - Duration: {format_time_duration(video_duration)} ({video_duration:.1f}s)
    - Description: "{(video_info.get('description') or '')[:300]}..."

    PODCAST REQUIREMENTS:
    1. Create a 7-10 minute podcast script (approximately 1000-1200 words) in conversation format.
    2. The hosts should have distinct personalities:
    - {host1_name}: {host1_personality}
    - {host2_name}: {host2_personality}
    3. The conversation must feel natural, with authentic dialogue, varied sentence lengths, and smooth transitions.
    4. Include an engaging introduction that hooks the listener, 3-5 clearly defined main discussion points, and a satisfying conclusion.
    5. The hosts should engage with each other (asking thoughtful questions, building on ideas, occasionally having friendly disagreements).
    6. {user_preferences}
    7. Don't reference "the video" or "the speaker in the video" - the hosts should discuss the topic as their own original conversation.
    8. Include occasional humor, personal anecdotes, and relatable examples to keep the discussion engaging.
    9. Format the script clearly with speaker names followed by their dialogue.
    10. Structure the conversation to have a clear beginning, middle, and end, with natural transitions between topics.

    TRANSCRIPT CONTENT TO DISCUSS:
    {full_text[:12000]}

    RETURN FORMAT:
    Please provide the podcast script in the following JSON format:
    {{
    "title": "Engaging podcast title",
    "description": "Brief description of the podcast episode",
    "hosts": ["{host1_name}", "{host2_name}"],
    "estimated_duration_minutes": X,
    "script": [
        {{
        "speaker": "{host1_name}", 
        "text": "Welcome to the show..."
        }},
        {{
        "speaker": "{host2_name}",
        "text": "Great to be here..."
        }},
        ...
    ]
    }}
    """
    
    try:
        logger.info("Generating podcast script with Gemini...")
        
        # Add language instruction to the prompt if needed
        language_instruction = ""
        if detected_language and detected_language != "en":
            language_instruction = f"Please respond in {detected_language} language. "
        enhanced_prompt = f"{language_instruction}{prompt}"
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        
        # Run Gemini API call in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                model.generate_content,
                enhanced_prompt
            )
        )
        
        end_time = time.time()
        logger.info(f"Time taken to generate podcast script: {end_time - start_time:.4f} seconds")
        
        raw_content = response.text
        
        # Extract JSON from response
        podcast_data = None
        
        # Try direct JSON parsing
        try:
            podcast_data = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed, trying regex extraction...")
            
            # Try regex extraction
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*\})', raw_content.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1) or json_match.group(2)
                    podcast_data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Regex JSON extraction failed")
            
            # If still no valid JSON, try to fix common issues
            if not podcast_data:
                logger.info("Trying to fix and extract JSON...")
                # Remove markdown code blocks
                cleaned = re.sub(r'```json\s*|\s*```', '', raw_content)
                # Try to find object bounds
                start_idx = cleaned.find('{')
                end_idx = cleaned.rfind('}')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    try:
                        json_str = cleaned[start_idx:end_idx+1]
                        podcast_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Fixed JSON extraction failed")
        
        # If JSON parsing fails, create a structured format manually
        if not podcast_data or not isinstance(podcast_data, dict) or "script" not in podcast_data:
            logger.warning("Failed to parse podcast data, creating structured format manually")
            
            # Extract title if possible
            title_match = re.search(r'"title":\s*"([^"]+)"', raw_content)
            podcast_title = title_match.group(1) if title_match else f"Podcast about {video_info.get('title', 'Video Topic')}"
            
            # Split by speaker indicators
            script_lines = []
            current_speaker = None
            current_text = []
            
            for line in raw_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                speaker_match = re.match(r'^({}|{}):\s*(.+)'.format(host1_name, host2_name), line)
                if speaker_match:
                    # Save previous speaker's text if exists
                    if current_speaker and current_text:
                        script_lines.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text)
                        })
                        current_text = []
                    
                    # Start new speaker
                    current_speaker = speaker_match.group(1)
                    current_text.append(speaker_match.group(2))
                elif current_speaker:
                    # Continue current speaker's text
                    current_text.append(line)
            
            # Add final speaker text
            if current_speaker and current_text:
                script_lines.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            # Create structured data
            podcast_data = {
                "title": podcast_title,
                "description": f"A conversation about {video_info.get('title', 'this interesting topic')}",
                "hosts": [host1_name, host2_name],
                "estimated_duration_minutes": 5,
                "script": script_lines if script_lines else [
                    {"speaker": host1_name, "text": f"Welcome to our discussion about {video_info.get('title', 'this topic')}!"},
                    {"speaker": host2_name, "text": "I'm excited to dive into this with you today."}
                ]
            }
        
        # Cache the result
        if len(_podcast_script_cache) >= _cache_size_limit:
            # Remove an arbitrary entry if cache is full
            oldest_key = next(iter(_podcast_script_cache))
            del _podcast_script_cache[oldest_key]
        
        _podcast_script_cache[cache_key] = podcast_data
        
        return podcast_data
    
    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        return {
            "error": f"Failed to generate podcast: {str(e)}",
            "title": f"Podcast about {video_info.get('title', 'Video Topic')}",
            "hosts": [host1_name, host2_name],
            "script": [
                {"speaker": host1_name, "text": f"Welcome to our discussion about {video_info.get('title', 'this topic')}!"},
                {"speaker": host2_name, "text": "Unfortunately, we're experiencing some technical difficulties with our script generation."}
            ]
        }

async def save_podcast_script(podcast_data, output_dir=OUTPUT_DIR):
    """
    Save the podcast script to a text file.
    """
    try:
        # Create podcast directory if it doesn't exist
        podcast_dir = os.path.join(output_dir, "podcast")
        os.makedirs(podcast_dir, exist_ok=True)
        
        # Sanitize podcast title for filename
        title = podcast_data.get('title', 'podcast')
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
        # Create the script file
        script_path = os.path.join(podcast_dir, f"{safe_title}.txt")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            # Write podcast information
            f.write(f"Title: {podcast_data.get('title', 'Untitled Podcast')}\n")
            f.write(f"Description: {podcast_data.get('description', '')}\n")
            f.write(f"Hosts: {', '.join(podcast_data.get('hosts', ['Host1', 'Host2']))}\n")
            f.write(f"Estimated Duration: {podcast_data.get('estimated_duration_minutes', 5)} minutes\n\n")
            
            # Write script content
            f.write("===== PODCAST SCRIPT =====\n\n")
            
            for line in podcast_data.get('script', []):
                speaker = line.get('speaker', 'Speaker')
                text = line.get('text', '')
                f.write(f"{speaker}: {text}\n\n")
        
        logger.info(f"Podcast script saved to: {script_path}")
        return script_path
        
    except Exception as e:
        logger.error(f"Error saving podcast script: {e}")
        return None