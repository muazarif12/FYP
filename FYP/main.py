import os
import re
from constants import OUTPUT_DIR
from downloader import download_video
from transcriber import get_youtube_transcript, transcribe_video
from summarizer import generate_key_moments_algorithmically, generate_response_async, get_welcome_message, generate_key_moments_with_titles, generate_enhanced_response
from highlights import extract_highlights, generate_highlights, generate_custom_highlights, merge_clips
from retrieval import initialize_indexes, retrieve_chunks
from utils import format_time_duration, format_timestamp
import asyncio
import time
from meeting_minutes import generate_meeting_minutes, save_meeting_minutes, format_minutes_for_display
from dubbing import create_english_dub
from video_qa import answer_video_question
from study_guide import generate_faq
from subtitling import adjust_subtitle_timing_by_offset, create_english_subtitles, embed_subtitles_in_video
from algorithmic_highlights import generate_highlights_algorithmically
import tkinter as tk
from tkinter import filedialog


def open_file_dialog():
    """
    Opens a file dialog for the user to select a video file.
    
    Returns:
        str or None: Path to the selected file, or None if canceled
    """
    # Create a hidden root window for the dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Define valid video file types
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.3gp"),
        ("MP4 files", "*.mp4"),
        ("AVI files", "*.avi"),
        ("MOV files", "*.mov"),
        ("MKV files", "*.mkv"),
        ("All files", "*.*")
    ]
    
    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=filetypes
    )
    
    # Destroy the root window after selection
    root.destroy()
    
    # Return the selected file path (or None if canceled)
    return file_path if file_path else None



def validate_video_file(file_path):
    """
    Validate that the provided path points to a valid video file.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is an error message or None
    """
    import os
    import mimetypes
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        return False, f"Not a file: {file_path}"
    
    # Check file size (limit to 2GB for example)
    max_size = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return False, f"File too large: {file_size/1024/1024:.1f}MB (max {max_size/1024/1024:.1f}MB)"
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.3gp']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        return False, f"Unsupported file format: {ext}. Supported formats: {', '.join(valid_extensions)}"
    
    # Try to get the mime type (optional, as mimetypes might not be accurate for all video formats)
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and not mime_type.startswith('video/'):
            return False, f"Not a video file. Detected type: {mime_type}"
    except:
        # If mime type check fails, just continue with extension validation
        pass
        
    return True, None

async def handle_file_upload():
    """
    Process a locally uploaded video file using a file dialog.
    
    Returns:
        Tuple containing (downloaded_file, video_title, video_description, video_id)
    """
    # Open file dialog to select video file
    file_path = open_file_dialog()
    
    # If user canceled the file dialog
    if not file_path:
        print("File selection canceled.")
        return None, None, None, None
    
    # Validate the video file
    is_valid, error_message = validate_video_file(file_path)
    if not is_valid:
        print(f"Error: {error_message}")
        return None, None, None, None
    
    # Generate a unique filename to avoid conflicts
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    
    # Use the existing file extension or default to .mp4
    if not ext:
        ext = ".mp4"
    
    # Create a destination path in the OUTPUT_DIR
    output_file = os.path.join(OUTPUT_DIR, f"uploaded_video{ext}")
    
    try:
        # Copy the file to the output directory
        print(f"Copying file to processing directory...")
        import shutil
        shutil.copy2(file_path, output_file)
        print(f"File copied successfully.")
    except Exception as e:
        print(f"Error copying file: {e}")
        return None, None, None, None
    
    # For uploaded files, we don't have video_id, but we need to return similar structure
    # as download_video function for compatibility
    video_title = f"Uploaded Video - {base_name}"
    video_description = "User uploaded video file"
    video_id = None
    
    return output_file, video_title, video_description, video_id

async def chatbot():
    start_time = time.time()  # Start time for the entire process
    
    # Display welcome message with feature introduction
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                         Welcome to VidSense!                      ║
╚══════════════════════════════════════════════════════════════════╝

VidSense helps you extract insights from videos with these features:

📊 VIDEO ANALYSIS:
  • "summarize" → Get a concise summary of the video
  • "key moments" → See main chapters/sections with timestamps
  • "key topics" → Extract main ideas and concepts
  • "meeting minutes" → Generate structured meeting notes (great for recorded meetings)

🎬 CONTENT CREATION:
  • "highlights" → Generate video highlights
  • "highlights 3 minutes" → Create highlights of specific length
  • "reel" → Create a short clip for social media
  • "podcast" → Generate a conversational podcast about the video
  • "casual podcast" → Create a podcast with specific style
  • "english dub" → Create English-dubbed version of non-English videos
  • "english subtitles" → Add English subtitles to non-English video


💬 CONVERSATION:
  • Ask any question about the video content
  • "help" → Show all available commands
  • "quit" → Exit the application
        
💬 VIDEO Q&A:
  • Ask any specific question about the video content
  • "What did the speaker say about X?"
  • "When does the video mention Y?"
  • "Show me the part where Z happens"
  • "faq" → Generate common questions and answers
  • For text-only answers: "text only what did they say about X?"
          


Type "help" anytime to see this list again.
""")
    
    # Add option for video input method
    print("\nHow would you like to provide your video?")
    print("1. Upload a video file from your device")
    print("2. Paste a YouTube video link")
    
    # Input validation loop for user choice
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 for file upload or 2 for YouTube link.")
    
    downloaded_file = None
    video_title = None
    video_description = None
    video_id = None
    
    if choice == "1":
        # Handle file upload with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            print("\nPlease select your video file from the file explorer window that will open.")
            
            # Open file dialog to select video file
            downloaded_file, video_title, video_description, video_id = await handle_file_upload()
            
            if downloaded_file:
                break
            
            if attempt < max_attempts - 1:
                retry = input(f"Failed to process the video. You have {max_attempts - attempt - 1} more attempts. Try again? (y/n): ")
                if retry.lower() not in ["y", "yes"]:
                    print("Returning to input selection...")
                    return await chatbot()  # Restart the chatbot
        else:
            print("Maximum attempts reached.")
    else:
        # Handle YouTube link (existing functionality) with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            video_url = input("Please provide the YouTube video link: ")
            
            # Allow user to go back to the choice selection
            if video_url.lower() in ["back", "return", "cancel"]:
                print("Returning to input selection...")
                return await chatbot()  # Restart the chatbot
                
            # Basic validation for YouTube URL
            if "youtube.com" not in video_url and "youtu.be" not in video_url:
                print("Warning: This doesn't appear to be a YouTube link. Continuing anyway...")
            
            print("\nDownloading and analyzing the video...")
            downloaded_file, video_title, video_description, video_id = await download_video(video_url)
            
            if downloaded_file:
                break
            
            if attempt < max_attempts - 1:
                print(f"Failed to download the video. You have {max_attempts - attempt - 1} more attempts.")
                print("Type 'back' to return to input selection.")
            else:
                print("Maximum attempts reached.")
    
    if not downloaded_file:
        print("Failed to process the video. Exiting.")
        return

    # Try to get YouTube transcript first
    if video_id:
        transcript_segments, full_timestamped_transcript, detected_language = await get_youtube_transcript(video_id)
    else:
        transcript_segments, full_timestamped_transcript, detected_language = None, None, None

    # If YouTube transcript is not available, fall back to Whisper transcription
    if not transcript_segments:
        print("No YouTube transcript found. Transcribing video using Whisper...")
        transcript_segments, full_timestamped_transcript, detected_language = await transcribe_video(downloaded_file)

    # Pre-computing search indexes
    print("Pre-computing search indexes...")
    full_text = " ".join([seg[2] for seg in transcript_segments])
    await initialize_indexes(full_text)
    print("Search indexes ready.")

    # Store video info for highlights and podcast
    video_info = {
        "title": video_title,
        "description": video_description
    }

    # Display welcome message in detected language
    welcome_message = await get_welcome_message(detected_language)
    print(f"\n{welcome_message}")
    
    # Show feature overview after loading
    if detected_language == "en":
        feature_overview = """
Available Features:
------------------
✓ Video Summary - Get the main points quickly
✓ Key Moments - See major timestamps and what happens at each
✓ Highlights - Create a shorter version with the best parts
✓ Podcast - Generate a conversation between hosts discussing the video
✓ Reels - Create short-form content for social media
✓ Meeting Minutes - Generate professional notes from meetings
✓ Custom Questions - Ask anything about the video content
✓ Interactive Questions - Ask anything about the video content with the prefix Interactive Q/A or extract clips

Try saying "meeting minutes" to generate notes for a meeting recording!
"""
    else:
        # Translate feature overview to detected language
        translation_prompt = f"""Please translate the following to {detected_language}:

Available Features:
------------------
✓ Video Summary - Get the main points quickly
✓ Key Moments - See major timestamps and what happens at each
✓ Highlights - Create a shorter version with the best parts
✓ Podcast - Generate a conversation between hosts discussing the video
✓ Reels - Create short-form content for social media
✓ Meeting Minutes - Generate professional notes from meetings
✓ Custom Questions - Ask anything about the video content

Try saying "meeting minutes" to generate notes for a meeting recording!
"""
        feature_overview = await generate_response_async(translation_prompt)
    
    print(feature_overview)

    # Variables to store generated content (lazy loading)
    key_moments_data = None
    highlights_generated = False
    reel_generated = False
    podcast_generated = False
    podcast_data = None
    podcast_path = None
    meeting_minutes_generated = False
    meeting_minutes_data = None
    meeting_minutes_path = None

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            # Translate goodbye message based on detected language
            if detected_language == "en":
                goodbye_message = "Thank you for using VidSense! Goodbye and have a great day."
            else:
                goodbye_prompt = f"Please translate 'Thank you for using VidSense! Goodbye and have a great day.' to {detected_language} language."
                goodbye_message = await generate_response_async(goodbye_prompt)
            print(goodbye_message)
            break

        print("\nProcessing your query...")
        query_start_time = time.time()  # Start time for each query
        
        # Initialize variables outside any conditional blocks
        target_duration = None
        custom_podcast_prompt = None

        # Identify query type for better response formatting
        query_type = "general"
        
        # Check for podcast-related requests
        if re.search(r'podcast|conversation|discussion|dialogue|talk show|interview|audio', user_input.lower()):
            # Extract any specific instructions for podcast style
            style_match = re.search(r'(casual|funny|serious|educational|debate|friendly|professional|entertaining)', user_input.lower())
            format_match = re.search(r'style[:]?\s*(\w+)', user_input.lower())
            
            # Determine podcast style from the request
            if style_match:
                podcast_style = style_match.group(1)
                custom_podcast_prompt = f"Make the podcast {podcast_style} in tone and style."
                
            if format_match:
                podcast_format = format_match.group(1)
                if custom_podcast_prompt:
                    custom_podcast_prompt += f" Follow a {podcast_format} format."
                else:
                    custom_podcast_prompt = f"Follow a {podcast_format} format."
            
            query_type = "podcast"

        
# Add this to your query type detection section in the main loop:
        elif re.search(r'english subtitles|add subtitles|create subtitles|subtitle', user_input.lower()):
            query_type = "english_subtitles"

        elif re.search(r'meeting minutes|minutes|meeting notes|meeting summary', user_input.lower()):
            query_type = "meeting_minutes"

        elif re.search(r'english dub|dub|dubbing|voice over|translate audio|translate voice', user_input.lower()):
            query_type = "english_dub"
        
        # Add this near other query_type detection patterns in the while True loop
        elif re.search(r'^(extract\s+clips?|extracting\s+clips?|interactive(\s+q/?a)?|video\s+q/?a|create\s+clips?|find\s+clips?|clip\s+extract|get\s+clips?|show\s+clips?|video\s+answer|answer\s+with\s+clip)s?:?', user_input, re.IGNORECASE):
            query_type = "interactive_qa"
            # Extract the actual question by removing the trigger phrase
            actual_question = re.sub(r'^(extract\s+clips?|extracting\s+clips?|interactive(\s+q/?a)?|video\s+q/?a|create\s+clips?|find\s+clips?|clip\s+extract|get\s+clips?|show\s+clips?|video\s+answer|answer\s+with\s+clip)s?:?\s*', '', user_input, flags=re.IGNORECASE)


        # First check for highlight-related requests since they can overlap with other patterns
        elif re.search(r'highlight|best parts|important parts', user_input.lower()):
            # Extract duration information if explicitly specified
            duration_match = re.search(r'(\d+)\s*(minute|min|minutes|second|sec|seconds)', user_input.lower())

            if duration_match:
                amount = int(duration_match.group(1))
                unit = duration_match.group(2)
                if unit.startswith('minute') or unit.startswith('min'):
                    target_duration = amount * 60
                else:
                    target_duration = amount
                query_type = "custom_duration_highlights"
            # Check for custom instructions in the request
            elif re.search(r'(where|ensure|keep|include|focus on|select|show|add|include|take|at timestamp|first moment|last moment|beginning|end|intro|conclusion)', user_input.lower()):
                query_type = "custom_prompt_highlights"
            else:
                query_type = "highlights"
        # Then check other query types if not a highlight request
        elif any(term in user_input.lower() for term in ["timeline", "key moments", "chapters", "sections", "timestamps"]):
            query_type = "key_moments"
        elif any(term in user_input.lower() for term in ["summarize", "summary", "overview", "what is the video about"]):
            query_type = "summary"
        elif any(term in user_input.lower() for term in ["key topics", "main points", "main ideas", "central themes"]):
            query_type = "key_topics"
        elif re.search(r'at (\d{1,2}:?\d{1,2}:?\d{0,2})|timestamp|(\d{1,2}:\d{2})', user_input.lower()):
            query_type = "specific_timestamp"
        elif re.search(r'reel|short clip|tiktok|instagram|social media', user_input.lower()):
            query_type = "reel"
        elif user_input.lower() == "help":
            await show_help_message(detected_language)
            continue

        # Handle podcast generation
        if query_type == "podcast":
            print("Generating a podcast based on the video content. This may take a few minutes...")
            
            if podcast_generated and podcast_path:
                print("Using previously generated podcast.")
                print(f"Podcast available at: {podcast_path}")
                
                # Display script information
                if podcast_data:
                    hosts = podcast_data.get('hosts', ['Host1', 'Host2'])
                    script = podcast_data.get('script', [])
                    
                    print(f"\nPodcast title: {podcast_data.get('title', 'Untitled Podcast')}")
                    print(f"Hosts: {', '.join(hosts)}")
                    print(f"Number of dialogue lines: {len(script)}")
                    
                    # Show the first few lines of the script
                    print("\nPreview of the podcast script:")
                    for i, line in enumerate(script[:6]):  # Show first 6 lines
                        if i >= 6:
                            break
                        print(f"{line.get('speaker', 'Speaker')}: {line.get('text', '')[:100]}{'...' if len(line.get('text', '')) > 100 else ''}")
                    
                    if len(script) > 6:
                        print("... (script continues)")
            else:
                # Generate new podcast
                from podcast_integration import generate_podcast
                podcast_path, podcast_data = await generate_podcast(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    custom_prompt=custom_podcast_prompt,
                    detected_language=detected_language
                )
                
                if podcast_path:
                    podcast_generated = True
                    print(f"\nPodcast generated successfully!")
                    print(f"Saved to: {podcast_path}")
                    
                    # Display script preview
                    if podcast_data:
                        script = podcast_data.get('script', [])
                        print("\nPreview of the podcast script:")
                        for i, line in enumerate(script[:6]):  # Show first 6 lines
                            if i >= 6:
                                break
                            print(f"{line.get('speaker', 'Speaker')}: {line.get('text', '')[:100]}{'...' if len(line.get('text', '')) > 100 else ''}")
                        
                        if len(script) > 6:
                            print("... (script continues)")
                else:
                    print("Sorry, I couldn't generate a podcast for this video. Please try again.")


        # Add this where you handle other query types in the main.py while loop
        elif query_type == "interactive_qa":
            # Extract the actual question
            actual_question = re.sub(r'^(extract clips|interactive q/?a):\s*', '', user_input, flags=re.IGNORECASE)
            
            print(f"Processing interactive Q&A for: {actual_question}")
            
            # Call the answer_video_question function from video_qa.py
            qa_result = await answer_video_question(
                transcript_segments, 
                downloaded_file, 
                actual_question, 
                full_text=" ".join([seg[2] for seg in transcript_segments]),
                generate_clip=True
            )
            
            # Display the answer
            print(f"\nAnswer: {qa_result['answer']}")
            
            # Display timestamps
            if qa_result['formatted_timestamps']:
                print("\nRelevant parts of the video:")
                for ts in qa_result['formatted_timestamps']:
                    print(f"• {ts}")
            
            # Display clip information
            if qa_result['clip_path']:
                print(f"\nClip generated: {qa_result['clip_path']}")
                print(f"Clip title: {qa_result['clip_title']}")
            else:
                print("\nNo clip could be generated. The video may not contain relevant information.")

        elif query_type == "english_subtitles":
            print(f"Creating subtitles for your video. This may take some time...")
            
            # Call the subtitling function with automatic sync
            subtitled_video_path, subtitle_path, subtitling_stats = await create_english_subtitles(
                downloaded_file,
                transcript_segments,
                detected_language,
                OUTPUT_DIR
            )
            
            if subtitle_path and os.path.exists(subtitle_path):
                print("\nSubtitle file created successfully!")
                print(f"Subtitle file saved to: {subtitle_path}")
                
                if subtitled_video_path and os.path.exists(subtitled_video_path):
                    print("\nVideo with embedded subtitles created successfully!")
                    print(f"Saved to: {subtitled_video_path}")
                else:
                    print("\nNote: Could not embed subtitles directly. You can use the subtitle file with your media player.")
                
                # Display stats about the subtitling
                if isinstance(subtitling_stats, dict):
                    print("\nSubtitling Statistics:")
                    print(f"• Original Language: {subtitling_stats.get('original_language', detected_language)}")
                    print(f"• Segments Processed: {subtitling_stats.get('segments_processed', len(transcript_segments))}")
                    print(f"• Synchronization: {subtitling_stats.get('sync_delay', 'Automatic')}s")
                    print(f"• Processing Time: {subtitling_stats.get('processing_time', 'N/A')}")
                
                # Ask user to check synchronization
                sync_feedback_loop = True
                current_subtitle_path = subtitle_path
                current_video_path = subtitled_video_path
                
                while sync_feedback_loop:
                    print("\nPlease check if the subtitles are properly synchronized with the speech in the video.")
                    sync_feedback = input("Are the subtitles synchronized correctly? (yes/no): ").lower()
                    
                    if sync_feedback in ["yes", "y", "correct", "good", "fine", "ok", "okay"]:
                        print("Great! The subtitles are synchronized correctly.")
                        sync_feedback_loop = False
                    
                    elif sync_feedback in ["no", "n", "wrong", "bad", "off", "incorrect"]:
                        print("\nLet's adjust the subtitle timing to make them match the speech better.")
                        
                        # Get adjustment direction and amount
                        adjustment_input = input("\nPlease tell me if the subtitles appear too early or too late, and by how many seconds (e.g., 'early by 2.5 seconds' or 'late by 4 seconds'): ")
                        
                        # Parse adjustment input
                        early_match = re.search(r'(early|before|ahead|soon|fast|quick)\s+by\s+(\d+\.?\d*)', adjustment_input.lower())
                        late_match = re.search(r'(late|after|behind|delayed|slow)\s+by\s+(\d+\.?\d*)', adjustment_input.lower())
                        
                        adjustment_seconds = 0
                        
                        if early_match:
                            # Subtitles appear too early, need to delay them
                            adjustment_seconds = float(early_match.group(2))
                            print(f"I'll delay the subtitles by {adjustment_seconds} seconds.")
                        elif late_match:
                            # Subtitles appear too late, need to advance them
                            adjustment_seconds = -float(late_match.group(2))
                            print(f"I'll advance the subtitles by {abs(adjustment_seconds)} seconds.")
                        else:
                            # Try to parse a direct seconds value
                            direct_seconds_match = re.search(r'(-?\d+\.?\d*)\s*(?:s|sec|seconds)?', adjustment_input)
                            if direct_seconds_match:
                                adjustment_seconds = float(direct_seconds_match.group(1))
                                if "early" in adjustment_input.lower():
                                    adjustment_seconds = abs(adjustment_seconds)  # Make sure it's positive
                                elif "late" in adjustment_input.lower():
                                    adjustment_seconds = -abs(adjustment_seconds)  # Make sure it's negative
                                    
                                if adjustment_seconds > 0:
                                    print(f"I'll delay the subtitles by {adjustment_seconds} seconds.")
                                else:
                                    print(f"I'll advance the subtitles by {abs(adjustment_seconds)} seconds.")
                            else:
                                print("I couldn't understand the adjustment needed. Please try again with a format like 'early by 2.5 seconds' or 'late by 4 seconds'.")
                                continue
                        
                        print("\nAdjusting subtitle timing...")
                        # Adjust the subtitle file
                        adjusted_subtitle_path = await adjust_subtitle_timing_by_offset(
                            current_subtitle_path, 
                            adjustment_seconds,
                            output_dir=os.path.dirname(current_subtitle_path)
                        )
                        
                        if adjusted_subtitle_path and os.path.exists(adjusted_subtitle_path):
                            print(f"Adjusted subtitle file created at: {adjusted_subtitle_path}")
                            
                            # Create new video with adjusted subtitles
                            print("Embedding adjusted subtitles into video...")
                            adjusted_video_path = await embed_subtitles_in_video(
                                downloaded_file,  # Use original video
                                adjusted_subtitle_path,
                                OUTPUT_DIR
                            )
                            
                            if adjusted_video_path and os.path.exists(adjusted_video_path):
                                print(f"New video with adjusted subtitles created at: {adjusted_video_path}")
                                current_subtitle_path = adjusted_subtitle_path
                                current_video_path = adjusted_video_path
                            else:
                                print("Failed to create new video, but you can use the adjusted subtitle file with your media player.")
                                current_subtitle_path = adjusted_subtitle_path
                        else:
                            print("Failed to adjust subtitle timing. Please try again with different values.")
                    else:
                        print("I didn't understand your response. Please answer 'yes' if the synchronization is correct or 'no' if it needs adjustment.")
            else:
                print(f"Sorry, I couldn't create subtitles for this video.")
                if isinstance(subtitling_stats, str):
                    print(f"Reason: {subtitling_stats}")

        # Handle key moments/timeline requests
        elif query_type == "key_moments":
            # Generate key moments only when requested (lazy loading)
            if key_moments_data is None:
                print("Analyzing key moments in the video...")
                # key_moments_structured, key_moments_formatted = await generate_key_moments_with_titles(
                #     transcript_segments, 
                #     full_timestamped_transcript, 
                #     detected_language
                # )
                key_moments_structured, key_moments_formatted = await generate_key_moments_algorithmically(
                    transcript_segments, 
                    full_timestamped_transcript
                )    

                key_moments_data = {
                    'structured': key_moments_structured,
                    'formatted': key_moments_formatted
                }

            # Create a response prefix in the detected language
            if detected_language == "en":
                response_prefix = "Here are the key moments in the video:"
            else:
                prefix_prompt = f"Please translate 'Here are the key moments in the video:' to {detected_language} language."
                response_prefix = await generate_response_async(prefix_prompt)

            print(f"\n{response_prefix}")
            print(key_moments_data['formatted'])
            
            # Display extra stats for key moments
            num_moments = len(key_moments_data['structured'])
            print(f"\nTotal key moments identified: {num_moments}")

        elif query_type in ["highlights", "custom_duration_highlights", "custom_prompt_highlights"]:
            # Show appropriate message based on highlight type
            if query_type == "custom_duration_highlights" and target_duration:
                unit = "minutes" if target_duration >= 60 else "seconds"
                amount = target_duration / 60 if target_duration >= 60 else target_duration
                print(f"Generating {amount} {unit} highlights...")
            elif query_type == "custom_prompt_highlights":
                print("Creating custom highlights based on your specific requirements...")
            else:
                print("Generating video highlights. This may take a few minutes...")

            # Check if user requested fast generation explicitly
            use_fast_method = ("fast" in user_input.lower() or "quick" in user_input.lower() or 
                            "algorithmic" in user_input.lower())
            
            # For custom prompts, always use LLM method since it understands specific requirements better
            if query_type == "custom_prompt_highlights":
                # Generate highlights with custom instructions
                highlight_segments = await generate_custom_highlights(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    user_input,
                    target_duration=target_duration
                )
                
                # Extract and merge clips for all highlight types
                if highlight_segments:
                    print("Extracting highlight clips...")
                    clip_paths, highlight_info = extract_highlights(downloaded_file, highlight_segments)
                    
                    print("Merging clips into final video...")
                    highlights_path = merge_clips(clip_paths, highlight_info, is_reel=False)
                else:
                    highlights_path = None
            
            else:  # Regular highlights or duration-specific highlights
                if use_fast_method:
                    print("Using fast algorithmic highlight generation...")
                    highlights_path, highlight_segments = await generate_highlights_algorithmically(
                        downloaded_file,
                        transcript_segments,
                        video_info,
                        target_duration=target_duration,
                        is_reel=False
                    )
                else:
                    # Use LLM-based method (original code)
                    _, highlight_segments = await generate_highlights(
                        downloaded_file,
                        transcript_segments,
                        video_info,
                        target_duration=target_duration,
                        is_reel=False
                    )
                    
                    # Extract and merge clips
                    if highlight_segments:
                        print("Extracting highlight clips...")
                        clip_paths, highlight_info = extract_highlights(downloaded_file, highlight_segments)
                        
                        print("Merging clips into final video...")
                        highlights_path = merge_clips(clip_paths, highlight_info, is_reel=False)
                    else:
                        highlights_path = None

        elif query_type == "reel":
            print("Generating a short reel for social media. This may take a few minutes...")

            # Check if user requested fast generation
            use_fast_method = ("fast" in user_input.lower() or "quick" in user_input.lower() or 
                            "algorithmic" in user_input.lower())
                            
            if use_fast_method:
                print("Using fast algorithmic reel generation...")
                reel_path, reel_segments = await generate_highlights_algorithmically(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    target_duration=60,  # Default to 60 seconds max for reels
                    is_reel=True
                )
            else:
                # Generate reel using LLM method (original code)
                reel_path, reel_segments = await generate_highlights(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    target_duration=60,  # Default to 60 seconds max for reels
                    is_reel=True
                )
            if reel_path:
                reel_generated = True
                # Format reel descriptions
                reel_descriptions = []
                total_duration = 0
                for hl in reel_segments:
                    start_time = format_timestamp(hl["start"])
                    end_time = format_timestamp(hl["end"])
                    duration = hl["end"] - hl["start"]
                    total_duration += duration
                    reel_descriptions.append(f"• {start_time} - {end_time} ({duration:.1f}s): {hl['description']}")

                # Display results
                print("\nReel generated successfully!")
                print(f"Total duration: {total_duration:.1f} seconds")
                print(f"Saved to: {reel_path}")
                print("\nReel segments:")
                for desc in reel_descriptions:
                    print(f"• {desc}")
            else:
                print("Sorry, I couldn't generate a reel for this video. Please try again.")

        else:
            # Retrieve relevant chunks for the query
            retrieved_docs = await retrieve_chunks(" ".join([seg[2] for seg in transcript_segments]), user_input, k=3)
            references = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Generate enhanced response based on query type
            response = await generate_enhanced_response(query_type, references, user_input, detected_language)
            print(f"\n{response}")

        query_end_time = time.time()  # End time for each query
        print(f"Time taken for the query: {query_end_time - query_start_time:.4f} seconds")

    end_time = time.time()  # End time for entire process
    print(f"\nTotal time taken for the entire process: {end_time - start_time:.4f} seconds")

# Update the show_help_message function to include the text-only option
async def show_help_message(detected_language="en", user_language_preference="en"):
    if user_language_preference == "en":
        help_message = """
VidSense Commands:
- "summarize" or "summary": Get a summary of the video content
- "key moments" or "timeline": See the main chapters/sections of the video
- "meeting minutes": Generate structured meeting notes and action items
- "highlights": Generate video highlights
- "highlights X minutes": Generate X minutes of highlights
- "fast highlights": Generate highlights more quickly (algorithmic method)
- "podcast": Generate a conversation-style podcast about the video content
- "podcast [style]": Generate a podcast with a specific style (casual, educational, etc.)
- "reel": Create a short clip for social media
- "english dub": Create English-dubbed version of non-English videos
- "english subtitles": Add English subtitles to non-English video
- "faq": Generate frequently asked questions about the video
- Ask specific questions like "What did they say about X?"
- "Interactive Q/A: your question" or "Extract clips: your question": Ask a question and get an answer with relevant video clips
- For text-only answers: Add "text only" before your question
- "help": Show this help message
- "quit": Exit the application
        """
    else:
        help_prompt = f"""
Please translate the following help message to {user_language_preference} language:

VidSense Commands:
- "summarize" or "summary": Get a summary of the video content
- "key moments" or "timeline": See the main chapters/sections of the video
- "meeting minutes": Generate structured meeting notes and action items
- "highlights": Generate video highlights
- "highlights X minutes": Generate X minutes of highlights
- "podcast": Generate a conversation-style podcast about the video content
- "podcast [style]": Generate a podcast with a specific style (casual, educational, etc.)
- "reel": Create a short clip for social media
- "english dub": Create English-dubbed version of non-English videos
- "faq": Generate frequently asked questions about the video
- Ask specific questions like "What did they say about X?" (creates answer clip)
- For text-only answers: Add "text only" before your question
- "help": Show this help message
- "quit": Exit the application
        """
        help_message = await generate_response_async(help_prompt, user_language_preference=user_language_preference)
        
    print(help_message)

if __name__ == "__main__":
    asyncio.run(chatbot())