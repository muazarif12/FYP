import os
import requests
from pydub import AudioSegment
from time import sleep

# === CONFIG ===
API_KEY = "sk_11e6fcf5d16cfc8ba38904e1ca673618e009eb190475a13a"  # 🔐 Replace with your actual key
VOICE_IDS = {
    "Alex": "EXAVITQu4vr4xnSDxMaL",  # Replace with your actual voice ID
    "Sam": "21m00Tcm4TlvDq8ikWAM"     # Replace with your actual voice ID
}

OUTPUT_DIR = "elevenlabs_output"
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "segments")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")

# Create necessary folders
os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
summary =  """In this episode of the AI Show, host Seth welcomes Ashley Yeo, Senior Program Manager at Microsoft’s Cognitive Services team, to discuss Text Analytics for Health—a powerful natural language processing (NLP) service designed to extract structured insights from unstructured medical text.
        Ashley shares the journey of launching the service, originally announced in July after being fast-tracked due to the pandemic. The service processes healthcare documents like doctor’s notes, discharge summaries, and research articles using advanced NLP techniques. It identifies medical entities (diagnoses, medications, symptoms, etc.), links them to standardized vocabularies (ICD-10, SNOMED, RxNorm), extracts relationships (like measurements and attributes), and even detects negations (e.g., "no weight loss").

        Unlike traditional text analytics, this isn’t just a repurposing—it's an extension developed in collaboration with Microsoft Research’s Health Next team and annotated by medical professionals.

        Ashley demos both a local container version and a new hosted API (soon to be asynchronous) using Postman. He emphasizes privacy and HIPAA compliance, explaining that no user data is stored—only metrics like document count are logged, and results expire after 48 hours.

        To learn more or request access, visit the Azure Cognitive Services documentation under Text Analytics for Health.
"""

prompt = f"""
You're writing a podcast script. Based on the following summary, generate a casual and engaging conversation between Alex and Sam. Format it as a Python list of (speaker, line) tuples.

Summary:
{summary}

Make the conversation natural, back-and-forth, and informative.
Return only the Python list, like this:
[("Alex", "Line..."), ("Sam", "Line...")]
"""

# === Conversation Script ===
conversation = [
    ("Alex", "Hey Sam, did you catch the latest episode of the AI Show?"),
    ("Sam", "Yeah! The one with Ashley Yeo talking about Text Analytics for Health? Super cool stuff."),
    ("Alex", "Right? I was blown away by how they're using NLP to dig into unstructured medical text."),
    ("Sam", "Exactly. Stuff like doctor’s notes, discharge summaries, even research papers. It pulls out things like diagnoses, medications, and symptoms."),
    ("Alex", "And it doesn’t just stop there—it actually links all that to standardized vocabularies like ICD-10, SNOMED, and RxNorm."),
    ("Sam", "Yeah, I thought that was clever. Plus, it detects relationships too, like measurements and attributes, and even knows when something’s a negation."),
    ("Alex", "You mean like when a note says 'no fever' or 'denies chest pain'?"),
    ("Sam", "Exactly! It catches that nuance, which is huge in medical contexts."),
    ("Alex", "What I loved was how Ashley explained it’s not just a rehash of existing tools—it’s something they built specifically for healthcare."),
    ("Sam", "Totally. And it was fast-tracked because of the pandemic, right? That gave it some serious urgency."),
    ("Alex", "Yup, and it’s been developed in collaboration with Microsoft Research’s Health Next team, with actual medical professionals helping with annotations."),
    ("Sam", "The demo part was sweet too. Ashley showed how it runs both in a local container and through a hosted API using Postman."),
    ("Alex", "And soon to be asynchronous, which is a game-changer for large-scale processing."),
    ("Sam", "Oh, and I appreciated how they emphasized privacy. No user data is stored, just some usage metrics like document counts."),
    ("Alex", "And results expire after 48 hours, so it stays compliant with HIPAA and other standards."),
    ("Sam", "If anyone’s curious, they can check out the Azure Cognitive Services docs—just look under Text Analytics for Health."),
    ("Alex", "Seriously, if you're working with medical data, this tool sounds like a lifesaver."),
    ("Sam", "No doubt. Can’t wait to see how it evolves with the asynchronous rollout."),
]

# === ElevenLabs Audio Generation Function ===
def generate_elevenlabs_audio(text, voice_id, filename):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.75
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Generated: {filename}")
            return True
        else:
            print(f"❌ Failed to generate {filename}")
            print(f"Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception during generation: {e}")
        return False

# === Generate Audio Segments ===
for i, (speaker, line) in enumerate(conversation):
    output_file = os.path.join(SEGMENT_DIR, f"{i}_{speaker}.mp3")
    if os.path.exists(output_file):
        print(f"⏩ Already exists: {output_file}")
        continue
    print(f"🎙️ Generating: {output_file}")
    success = generate_elevenlabs_audio(line, VOICE_IDS[speaker], output_file)
    if not success:
        print(f"⚠️ Skipping failed segment: {output_file}")
    sleep(1)  # prevent rate limiting

# === Combine Segments ===
segments = []
for i, (speaker, _) in enumerate(conversation):
    segment_path = os.path.join(SEGMENT_DIR, f"{i}_{speaker}.mp3")
    if not os.path.exists(segment_path):
        print(f"⚠️ Skipping missing segment: {segment_path}")
        continue
    try:
        segment = AudioSegment.from_file(segment_path)
        segments.append(segment + AudioSegment.silent(duration=500))  # 0.5s pause
    except Exception as e:
        print(f"⚠️ Error loading segment {segment_path}: {e}")
        continue

if segments:
    final_audio = sum(segments)
    final_audio_path = os.path.join(FINAL_DIR, "final_podcast.mp3")
    final_audio.export(final_audio_path, format="mp3")
    print(f"\n✅ Podcast generated at: {final_audio_path}")
else:
    print("❌ No segments available to combine.")

