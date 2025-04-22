
import asyncio, time
from pytubefix import YouTube


async def download_video(link):
    try:
        start_time = time.time()  # Start time for download
        yt = YouTube(link)
        
        # Extract video ID from the YouTube URL
        video_id = None
        if "youtube.com" in link or "youtu.be" in link:
            if "youtube.com/watch" in link:
                video_id = link.split("v=")[1].split("&")[0]
            elif "youtu.be" in link:
                video_id = link.split("/")[-1].split("?")[0]
        
        stream = yt.streams.get_highest_resolution()
        ext = stream.mime_type.split("/")[-1]
        filename = f"downloaded_video.{ext}"
        await asyncio.to_thread(stream.download, filename=filename)  # Run in a separate thread
        end_time = time.time()  # End time for download
        print(f"Video downloaded as {filename}")

        print(f"Time taken for video download: {end_time - start_time:.4f} seconds")
        return filename, yt.title, yt.description, video_id
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None, None, None