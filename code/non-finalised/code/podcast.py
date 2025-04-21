import pyttsx3
from pydub import AudioSegment
import os

# Initialize engine
engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')
for idx, voice in enumerate(voices):
    print(f"{idx}: {voice.name} - {voice.id}")

# Assign different voices to Alex and Sam
alex_voice = voices[0].id  # e.g., male
sam_voice = voices[1].id   # e.g., female

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


os.makedirs("pyttsx_output/segments", exist_ok=True)
os.makedirs("pyttsx_output/final", exist_ok=True)


segments = []

for i, (speaker, text) in enumerate(conversation):
    filename = f"pyttsx_output/segments/{i}_{speaker}.mp3"
    engine.setProperty('voice', alex_voice if speaker == "Alex" else sam_voice)
    engine.save_to_file(text, filename)

# This is important! It triggers the processing of all queued lines
engine.runAndWait()

# Combine using pydub (after audio is saved)
for i, (speaker, _) in enumerate(conversation):
    segment = AudioSegment.from_file(f"pyttsx_output/segments/{i}_{speaker}.mp3")
    segments.append(segment + AudioSegment.silent(duration=500))  # pause between lines

# Final podcast output
final = sum(segments)
final.export("pyttsx_output/final/final_podcast.mp3", format="mp3")
