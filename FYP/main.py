import re
from downloader import download_video
from transcriber import get_youtube_transcript, transcribe_video
from summarizer import generate_response_async, get_welcome_message, generate_key_moments_with_titles, generate_enhanced_response
from highlights import extract_highlights, generate_highlights, generate_custom_highlights, merge_clips
from retrieval import retrieve_chunks
from utils import format_timestamp
import asyncio
import time


async def chatbot():
    start_time = time.time()  # Start time for the entire process
    print("Welcome to VidSense! Type 'quit' to end the conversation.\n")
    video_url = input("Please provide the video link: ")

    # Run video download
    print("\nDownloading and analyzing the video...")
    downloaded_file, video_title, video_description, video_id = await download_video(video_url)
    if not downloaded_file:
        print("Failed to download the video. Exiting.")
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

    # Store video info for highlights
    video_info = {
        "title": video_title,
        "description": video_description
    }

    # Display welcome message in detected language
    welcome_message = await get_welcome_message(detected_language)
    print(f"\n{welcome_message}")

    # Variables to store generated content (lazy loading)
    key_moments_data = None
    highlights_generated = False
    reel_generated = False

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
        
        # Initialize target_duration outside any conditional blocks
        target_duration = None

        # Identify query type for better response formatting
        query_type = "general"
        
        # First check for highlight-related requests since they can overlap with other patterns
        if re.search(r'highlight|best parts|important parts|generate', user_input.lower()):
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
            elif re.search(r'(keep|include|focus on|select|show|add|include|take|at timestamp|first moment|last moment|beginning|end|intro|conclusion)', user_input.lower()):
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

        # Handle key moments/timeline requests
        if query_type == "key_moments":
            # Generate key moments only when requested (lazy loading)
            if key_moments_data is None:
                print("Analyzing key moments in the video...")
                key_moments_structured, key_moments_formatted = await generate_key_moments_with_titles(
                    transcript_segments, 
                    full_timestamped_transcript, 
                    detected_language
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

            # Generate highlight segments
            if query_type == "custom_prompt_highlights":
                # Generate highlights with custom instructions
                highlight_segments = await generate_custom_highlights(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    user_input,
                    target_duration=target_duration
                )
            else:  # Regular highlights or duration-specific highlights
                # Both use the same function, but with different target_duration values
                _, highlight_segments = await generate_highlights(
                    downloaded_file,
                    transcript_segments,
                    video_info,
                    target_duration=target_duration,
                    is_reel=False
                )

            # Extract and merge clips for all highlight types
            if highlight_segments:
                print("Extracting highlight clips...")
                clip_paths, highlight_info = extract_highlights(downloaded_file, highlight_segments)
                
                print("Merging clips into final video...")
                highlights_path = merge_clips(clip_paths, highlight_info, is_reel=False)
            else:
                highlights_path = None

            # Format and display results
            if highlights_path:
                highlights_generated = True
                # Format highlight descriptions
                highlight_descriptions = []
                total_duration = 0
                for hl in highlight_segments:
                    start_time = format_timestamp(hl["start"])
                    end_time = format_timestamp(hl["end"])
                    duration = hl["end"] - hl["start"]
                    total_duration += duration
                    highlight_descriptions.append(f"• {start_time} - {end_time} ({duration:.1f}s): {hl['description']}")

                # Display results
                print("\nHighlights generated successfully!")
                print(f"Total duration: {total_duration:.1f} seconds")
                print(f"Saved to: {highlights_path}")
                print("\nHighlight segments:")
                for desc in highlight_descriptions:
                    print(f"• {desc}")
            else:
                print("Sorry, I couldn't generate highlights for this video. Please try again.")

        # Handle reel generation
        elif query_type == "reel":
            print("Generating a short reel for social media. This may take a few minutes...")

            # Generate reel
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


if __name__ == "__main__":
    asyncio.run(chatbot())