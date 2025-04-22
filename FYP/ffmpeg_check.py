import subprocess
import os
import sys
from logger_config import logger

def check_ffmpeg_installed():
    """
    Check if ffmpeg is installed on the system and accessible in PATH.
    
    Returns:
        bool: True if ffmpeg is available, False otherwise
    """
    try:
        # Try to execute ffmpeg -version
        process = subprocess.Popen(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        # Check return code
        if process.returncode == 0:
            logger.info("FFmpeg is installed and accessible.")
            return True
        else:
            logger.warning("FFmpeg check returned non-zero exit code.")
            return False
            
    except Exception as e:
        logger.warning(f"FFmpeg check failed: {e}")
        return False

def ffmpeg_installation_guide():
    """
    Returns a guide for installing FFmpeg based on the user's operating system.
    
    Returns:
        str: Installation instructions for FFmpeg
    """
    if sys.platform.startswith('win'):
        return """
FFmpeg is required for audio generation but was not found on your system.

To install FFmpeg on Windows:
1. Download from https://ffmpeg.org/download.html (Windows builds)
2. Extract the zip file
3. Add the bin folder to your system PATH
4. Restart your terminal/console

Alternatively, you can install using a package manager:
- With Chocolatey: choco install ffmpeg
- With Scoop: scoop install ffmpeg
"""
    elif sys.platform.startswith('darwin'):
        return """
FFmpeg is required for audio generation but was not found on your system.

To install FFmpeg on macOS:
1. Using Homebrew: brew install ffmpeg
2. Using MacPorts: sudo port install ffmpeg

After installation, restart your terminal.
"""
    else:  # Linux and others
        return """
FFmpeg is required for audio generation but was not found on your system.

To install FFmpeg on Linux:
- Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg
- Fedora: sudo dnf install ffmpeg
- Arch Linux: sudo pacman -S ffmpeg
- CentOS/RHEL: sudo yum install epel-release && sudo yum install ffmpeg

After installation, restart your terminal.
"""

def verify_dependencies():
    """
    Verify all dependencies required for podcast audio generation.
    
    Returns:
        tuple: (bool, str) - (all dependencies installed, message)
    """
    missing_deps = []
    
    # Check for gtts
    try:
        import gtts
    except ImportError:
        missing_deps.append("gtts (pip install gtts)")
    
    # Check for edge-tts (optional)
    try:
        import edge_tts
    except ImportError:
        missing_deps.append("edge-tts (pip install edge-tts) - OPTIONAL for better voice differentiation")
    
    # Check for ffmpeg
    if not check_ffmpeg_installed():
        missing_deps.append("ffmpeg")
    
    if missing_deps:
        # If only edge-tts is missing, that's not critical
        if len(missing_deps) == 1 and "edge-tts" in missing_deps[0]:
            return True, "Note: Installing edge-tts would provide better voice differentiation."
        
        message = "Missing dependencies for podcast audio generation:\n- " + "\n- ".join(missing_deps)
        
        if "ffmpeg" in missing_deps:
            message += "\n\n" + ffmpeg_installation_guide()
            
        return False, message
    
    return True, "All required dependencies for podcast audio generation are installed."

if __name__ == "__main__":
    # Test the functions
    ffmpeg_available = check_ffmpeg_installed()
    print(f"FFmpeg available: {ffmpeg_available}")
    
    if not ffmpeg_available:
        print(ffmpeg_installation_guide())
    
    all_deps, message = verify_dependencies()
    print(message)