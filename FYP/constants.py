import os

OUTPUT_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Add these to your constants.py file if needed

# Maximum audio segment duration for dubbing (in seconds)
MAX_DUBBING_SEGMENT_DURATION = 15

# Supported languages for translation to English
SUPPORTED_LANGUAGES_FOR_DUBBING = [
    "ar", "zh", "fr", "de", "hi", "it", "ja", "ko", "pt", "ru", "es", "tr", "ur"
]

# Voice options for English dubbing
ENGLISH_DUBBING_VOICES = {
    "male_1": "en-US-GuyNeural",
    "male_2": "en-US-TonyNeural",
    "female_1": "en-US-JennyNeural",
    "female_2": "en-US-AriaNeural"
}
