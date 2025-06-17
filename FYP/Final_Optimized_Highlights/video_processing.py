from moviepy import VideoFileClip, concatenate_videoclips

def process_video(selected_chunks, video_file_path, output_path='output_video.mp4'):
    """
    Process video by extracting segments and merging them.
    
    Args:
        selected_chunks: List of chronologically sorted chunks to include in the video
        video_file_path: Path to the video file
        output_path: Path where the output video will be saved
    """
    print('Processing video...')
    
    # Load the video
    print(f"Loading video from {video_file_path}...")
    video = VideoFileClip(video_file_path)
    
    # Extract clips for each chunk
    clips = []
    for i, chunk in enumerate(selected_chunks):
        start_time = chunk.get('start', 0)
        end_time = chunk.get('end', 0)
        
        # Ensure we have valid start and end times
        if start_time >= end_time or start_time < 0 or end_time > video.duration:
            print(f"Warning: Invalid time range for chunk {i}: {start_time} - {end_time}")
            continue
        
        print(f"Extracting clip {i+1}/{len(selected_chunks)}: {start_time:.2f}s - {end_time:.2f}s")
        clip = video.subclipped(start_time, end_time)
        clips.append(clip)
    
    if not clips:
        print("No valid clips found!")
        return
    
    # Concatenate the clips
    print("Concatenating clips...")
    final_clip = concatenate_videoclips(clips)
    
    # Write the output video
    print(f"Writing output to {output_path}...")
    final_clip.write_videofile(output_path, codec='libx264')
    
    # Close all clips to free resources
    final_clip.close()
    for clip in clips:
        clip.close()
    video.close()
    
    print("Video processing completed!")