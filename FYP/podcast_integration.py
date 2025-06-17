# Import necessary podcast generation functions
from podcast_generator import generate_podcast_script, save_podcast_script
from gtts_audio_generator import generate_podcast_audio_with_gtts
from ffmpeg_check import verify_dependencies
import os

# Function to generate podcast with progress reporting
async def generate_podcast(downloaded_file, transcript_segments, video_info, custom_prompt=None, detected_language="en"):
    """
    Generate a podcast based on video content.
    
    Args:
        downloaded_file: Path to the downloaded video file
        transcript_segments: List of transcript segments
        video_info: Dictionary with video title and description
        custom_prompt: Optional custom instructions for podcast style
        detected_language: Language code for the video
    
    Returns:
        Tuple containing (podcast_path, podcast_data)
    """
    try:
        print("Generating podcast script from video content...")
        
        # Generate the podcast script
        podcast_data = await generate_podcast_script(
            transcript_segments, 
            video_info,
            detected_language,
            custom_prompt
        )
        
        if "error" in podcast_data:
            print(f"Error generating podcast script: {podcast_data['error']}")
            return None, podcast_data
        
        # Show script statistics
        num_lines = len(podcast_data.get('script', []))
        hosts = podcast_data.get('hosts', ['Host1', 'Host2'])
        estimated_duration = podcast_data.get('estimated_duration_minutes', 5)
        
        print(f"\nPodcast script generated successfully!")
        print(f"Title: {podcast_data.get('title', 'Untitled Podcast')}")
        print(f"Hosts: {', '.join(hosts)}")
        print(f"Lines of dialogue: {num_lines}")
        print(f"Estimated duration: {estimated_duration} minutes")
        
        # Save the script to a text file
        print("\nSaving podcast script to text file...")
        script_path = await save_podcast_script(podcast_data)
        
        if not script_path:
            print("Failed to save podcast script")
            return None, podcast_data
        
        print(f"Podcast script saved to: {script_path}")
        
        # Check dependencies before attempting audio generation
        deps_ok, deps_message = verify_dependencies()
        if not deps_ok:
            print("\nCannot generate audio due to missing dependencies:")
            print(deps_message)
            print("Podcast script is still available as a text file.")
            return script_path, podcast_data
            
        # Try to generate audio
        print("\nAttempting to generate podcast audio. This may take a few minutes...")
        try:
            # Get output directory from script path
            output_dir = os.path.dirname(os.path.dirname(script_path))
            
            # Generate audio using gtts
            audio_path = await generate_podcast_audio_with_gtts(
                podcast_data,
                output_dir,
                language=detected_language
            )
            
            if audio_path and os.path.exists(audio_path):
                print(f"\nPodcast audio generated successfully!")
                print(f"Saved to: {audio_path}")
                podcast_path = audio_path
            else:
                print("\nAudio generation failed. Text script is still available.")
                podcast_path = script_path
                
        except Exception as audio_error:
            print(f"\nAudio generation error: {audio_error}")
            print("Text script is still available.")
            podcast_path = script_path
        
        # Return the podcast path and data
        return podcast_path, podcast_data
        
    except Exception as e:
        print(f"Error generating podcast: {e}")
        return None, {"error": str(e)}