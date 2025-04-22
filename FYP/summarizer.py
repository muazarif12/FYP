import re
import time
import ollama
from utils import format_timestamp
import asyncio
import hashlib
import functools


# Response cache to store previously generated responses
_response_cache = {}
_cache_size_limit = 100  # Limit cache size to prevent memory issues

# Function to generate response using Ollama's chat functionality with improved prompting and caching
async def generate_response_async(prompt_text, language_code="en"):
    """
    Generate a response with caching to improve performance for repeated prompts.
    """
    # Create a cache key from the prompt and language
    cache_key = hashlib.md5(f"{prompt_text}:{language_code}".encode()).hexdigest()
    
    # Check if response is in cache
    if cache_key in _response_cache:
        print("Using cached response (saved LLM call)")
        return _response_cache[cache_key]
    
    model = "deepseek-r1:7b"  # Use any model you prefer
    
    try:
        # Add language instruction to the prompt
        language_instruction = ""
        if language_code and language_code != "en":
            language_instruction = f"Please respond in {language_code} language. "

        enhanced_prompt = f"{language_instruction}{prompt_text}"

        start_time = time.time()  # Start time for query

        # Run Ollama in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
        )

        end_time = time.time()  # End time for query
        print(f"Time taken for LLM query: {end_time - start_time:.4f} seconds")
        
        result = response['message']['content']
        
        # Cache the response
        if len(_response_cache) >= _cache_size_limit:
            # Remove oldest entry if cache is full
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        
        _response_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while generating the response."

# Get welcome message based on detected language with caching
_welcome_message_cache = {}

async def get_welcome_message(language_code):
    """Get welcome message with caching to improve performance."""
    if language_code in _welcome_message_cache:
        return _welcome_message_cache[language_code]
        
    if language_code == "en":
        message = "Welcome to VidSense! I've analyzed the video and I'm ready to answer your questions. You can ask me to summarize the video, explain key topics, provide timestamps, generate highlights, or answer specific questions about the content. Type 'quit' to end the conversation."
    else:
        prompt = f"Please translate the following message to {language_code} language: 'Welcome to VidSense! I've analyzed the video and I'm ready to answer your questions. You can ask me to summarize the video, explain key topics, provide timestamps, generate highlights, or answer specific questions about the content. Type 'quit' to end the conversation.'"
        message = await generate_response_async(prompt)
    
    _welcome_message_cache[language_code] = message
    return message
    

async def generate_key_moments_with_titles(transcript_segments, full_timestamped_transcript, language_code="en"):
    """
    Generate key moments with titles from transcript, and return them in a structured format.
    Return both the raw text format and a structured list for programmatic use.
    """
    # Extract actual timestamps from the full transcript
    actual_timestamps = []
    transcript_lines = full_timestamped_transcript.strip().split('\n')
    
    for line in transcript_lines:
        if re.match(r'\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:', line):
            # Extract just the start timestamp
            timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                actual_timestamps.append(timestamp)
    
    # Build a more precise prompt
    prompt = f"""
    You are VidSense, an advanced video summarizer with expertise in identifying key moments within video transcripts. Your task is to extract the most significant moments from this video and organize them into a clear, professional timeline.

    Your output should:

    1. Identify 6-10 key moments that mark important transitions, topics, or sections in the video.
    2. Ensure the first moment is the **Introduction**, starting at **00:00:00**.
    3. CRITICALLY IMPORTANT: Use ONLY timestamps that actually appear in the transcript. You MUST select from these exact timestamps:
       {', '.join(actual_timestamps)}
    4. Provide **clear, descriptive titles** for each key moment that accurately reflect the topic.
    5. For each key moment, include 1-2 sentences describing what happens or is discussed at that timestamp.

    IMPORTANT: Titles should be engaging and specific, avoiding generic descriptions.

    Here is the timestamped transcript to analyze:
    {full_timestamped_transcript}

    Please format your response in the following structured JSON format (and nothing else):

    [
        {{
            "timestamp": "00:00:00",
            "title": "Introduction",
            "description": "Brief description of what happens at this point"
        }},
        {{
            "timestamp": "HH:MM:SS",
            "title": "Clear Title for Key Moment 1",
            "description": "Brief description of what happens at this point"
        }},
        ... and so on
    ]

    REMEMBER: Only use timestamps that actually appear in the transcript. Do not invent or modify timestamps.
    """

    start_time = time.time()
    response = await generate_response_async(prompt, language_code)
    end_time = time.time()
    print(f"Time taken to generate key moments with titles: {end_time - start_time:.4f} seconds")

    # Try to parse the response as JSON
    key_moments_structured = []
    try:
        # Find JSON array in response
        json_match = re.search(r'\[\s*{.*}\s*\]', response.replace('\n', ' '), re.DOTALL)
        if json_match:
            import json
            key_moments_structured = json.loads(json_match.group(0))
        else:
            # If no JSON found, try to extract manually
            key_moments_structured = extract_key_moments_manually(response, actual_timestamps)
            
    except Exception as e:
        print(f"Error parsing key moments response: {e}")
        # Try again with stricter formatting if JSON parsing failed
        stricter_prompt = f"""
        ERROR: Your previous response could not be parsed correctly.
        
        Please generate key moments for the video transcript in VALID JSON FORMAT ONLY, with no other text.
        Use this exact format, and do not include any explanation or additional text:
        
        [
            {{
                "timestamp": "00:00:00",
                "title": "Introduction",
                "description": "Brief description of what happens at this point"
            }},
            ... and so on
        ]
        
        Use ONLY these exact timestamps that appear in the transcript:
        00:00:00 (for introduction)
        {', '.join(actual_timestamps)}
        
        Original transcript:
        {full_timestamped_transcript[:1000]}... (transcript truncated)
        """
        
        response = await generate_response_async(stricter_prompt, language_code)
        try:
            # Try to parse JSON again
            json_match = re.search(r'\[\s*{.*}\s*\]', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                key_moments_structured = json.loads(json_match.group(0))
            else:
                # If still no JSON, create a basic structure
                key_moments_structured = [{"timestamp": "00:00:00", "title": "Introduction", "description": "Beginning of the video."}]
                
        except Exception as inner_e:
            print(f"Error parsing key moments response (second attempt): {inner_e}")
            # Create a fallback structure
            key_moments_structured = [{"timestamp": "00:00:00", "title": "Introduction", "description": "Beginning of the video."}]
    
    # Generate a human-readable formatted text version
    formatted_text = format_key_moments(key_moments_structured, language_code)
    
    return key_moments_structured, formatted_text

def extract_key_moments_manually(text, valid_timestamps):
    """Extract key moments from text response when JSON parsing fails."""
    key_moments = []
    lines = text.strip().split('\n')
    current_timestamp = None
    current_title = None
    current_description = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to extract timestamp
        timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            # If we already have a timestamp, save the previous entry
            if current_timestamp:
                key_moments.append({
                    "timestamp": current_timestamp,
                    "title": current_title or "Unknown",
                    "description": current_description or ""
                })
                
            # Start a new entry
            current_timestamp = timestamp_match.group(1)
            title_match = re.search(r'\d{2}:\d{2}:\d{2}\s*-\s*(.*)', line)
            if title_match:
                current_title = title_match.group(1).strip()
            else:
                current_title = "Unknown"
            current_description = ""
        elif current_timestamp and not current_description:
            # This line is likely the description
            current_description = line
    
    # Add the last entry
    if current_timestamp:
        key_moments.append({
            "timestamp": current_timestamp,
            "title": current_title or "Unknown",
            "description": current_description or ""
        })
    
    # Ensure we have at least an introduction
    if not key_moments or key_moments[0]["timestamp"] != "00:00:00":
        key_moments.insert(0, {
            "timestamp": "00:00:00",
            "title": "Introduction",
            "description": "Beginning of the video."
        })
    
    return key_moments

def format_key_moments(key_moments_structured, language_code="en"):
    """Format structured key moments into a human-readable text format."""
    formatted_lines = []
    
    for moment in key_moments_structured:
        timestamp = moment.get("timestamp", "00:00:00")
        title = moment.get("title", "Unknown")
        description = moment.get("description", "")
        
        # Format similar to highlight segments
        formatted_line = f"• {timestamp} - {title}"
        if description:
            formatted_line += f" - {description}"
            
        formatted_lines.append(formatted_line)
    
    return "\n".join(formatted_lines)


# Enhanced response generation with caching
_enhanced_response_cache = {}

async def generate_enhanced_response(query_type, retrieval_data, user_input, detected_language="en"):
    """
    Generate enhanced responses with caching for repeated or similar queries.
    """
    # Create simplified retrieval data for caching (avoid caching exact retrieval text)
    # This allows caching even when retrieval data is slightly different 
    simplified_data = hashlib.md5(retrieval_data.encode()).hexdigest()
    
    # Create cache key that captures the essence of the query
    cache_key = f"{query_type}:{simplified_data[:10]}:{hashlib.md5(user_input.encode()).hexdigest()}:{detected_language}"
    
    # Check if we have a cached response
    if cache_key in _enhanced_response_cache:
        print("Using cached enhanced response")
        return _enhanced_response_cache[cache_key]
    
    # If not cached, proceed to generate
    if query_type == "summary":
        prompt = f"""
        You are VidSense, an intelligent video summarizer. Create a clear, concise summary of this video based on the retrieved information.

        The summary should be well-structured, engaging, and capture the main narrative of the video. Focus on the core message, key points, and overall flow.

        Write in a confident, professional tone as if you've watched the entire video yourself. Avoid using phrases like "based on the retrieved information" or "it appears that" - instead, speak with authority about the content.

        Retrieved video content:
        {retrieval_data}

        Create a summary that is 3-5 paragraphs long. Begin with an overview sentence, then develop the main points in the middle paragraphs, and end with a concluding thought.
        """
    elif query_type == "key_topics":
        prompt = f"""
        You are VidSense, an intelligent video analyzer. Based on the retrieved information, identify and explain the 4-6 most important topics or themes covered in this video.

        For each key topic:
        1. Provide a clear, concise title
        2. Write a brief explanation (2-3 sentences) that captures the essence of how this topic was presented in the video

        Present the information in a structured, easy-to-read format. Write with confidence and authority as if you've watched the entire video yourself.

        Retrieved video content:
        {retrieval_data}

        Focus only on the most significant topics that form the core of the video's message.
        """
    elif query_type == "specific_timestamp":
        # Extract any timestamp mentions from the query
        timestamp_pattern = r'(\d{1,2}:?\d{1,2}:?\d{0,2})'
        timestamps = re.findall(timestamp_pattern, user_input)
        timestamp_mention = timestamps[0] if timestamps else "the specified time"

        prompt = f"""
        You are VidSense, an intelligent video analyzer. The user is asking about content at or around {timestamp_mention} in the video.

        Based on the retrieved information, provide a detailed, accurate explanation of what was discussed at this specific point in the video. If the exact timestamp isn't in the retrieved information, focus on the closest relevant content.

        Write in a confident, knowledgeable tone as if you've watched the entire video yourself. Be specific about what was said, shown, or explained at this timestamp.

        User query: "{user_input}"

        Retrieved video content:
        {retrieval_data}

        Provide a clear, direct answer about the content at {timestamp_mention}, focusing on precisely what was covered at that moment in the video.
        """
    else:  # general query
        prompt = f"""
        You are VidSense, an intelligent video analyzer. Answer the user's question based on the content of the video.

        User query: "{user_input}"

        Retrieved video content:
        {retrieval_data}

        Instructions:
        1. Provide a clear, direct answer to the question based on the video content
        2. Include specific details, examples, or quotes from the video when relevant
        3. Write in a confident, knowledgeable tone as if you've watched the entire video yourself
        4. If the retrieved information doesn't contain enough details to fully answer the question, acknowledge this briefly but still provide the best possible answer based on what's available
        5. Keep your response concise and focused on answering exactly what was asked

        If the question relates to a specific moment or section of the video, mention the approximate timestamp or section where this information appears.
        """

    # Generate response
    response = await generate_response_async(prompt, detected_language)
    
    # Cache the response
    if len(_enhanced_response_cache) >= _cache_size_limit:
        # Remove an arbitrary entry if cache is full
        oldest_key = next(iter(_enhanced_response_cache))
        del _enhanced_response_cache[oldest_key]
    
    _enhanced_response_cache[cache_key] = response
    
    return response
