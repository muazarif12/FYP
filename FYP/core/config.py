import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "VidSense API"
    
    # Output directories
    OUTPUT_DIR: str = "downloads"
    HIGHLIGHTS_DIR: str = os.path.join(OUTPUT_DIR, "highlights")
    PODCAST_DIR: str = os.path.join(OUTPUT_DIR, "podcast")
    TRANSCRIPTS_DIR: str = os.path.join(OUTPUT_DIR, "transcripts")
    TEMP_DIR: str = os.path.join(OUTPUT_DIR, "temp")
    
    # File size limits (in MB)
    MAX_UPLOAD_SIZE: int = 1024  # 1GB
    
    # API rate limits
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Model configurations
    WHISPER_MODEL: str = "large-v3-turbo"  # or "small", "medium", "large"
    LLM_MODEL: str = "deepseek-r1:7b"
    
    # Feature flags
    ENABLE_YOUTUBE_API: bool = True
    ENABLE_VIDEO_UPLOAD: bool = True

    # ✅ Add this line to fix the error
    GOOGLE_API_KEY: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.HIGHLIGHTS_DIR, exist_ok=True)
os.makedirs(settings.PODCAST_DIR, exist_ok=True)
os.makedirs(settings.TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)