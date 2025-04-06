import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

import re

def parse_segments_from_file(filepath):
    segments = []
    current = {}
    print("parsing input file ...............................")
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith("id:"):
                current['id'] = [int(line.split(":", 1)[1].strip())]
            elif line.startswith("start:"):
                current["start"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("end:"):
                current["end"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("text:"):
                current["text"] = line.split(":", 1)[1].strip()
            elif line.startswith("tokens:"):
                current["tokens"] = eval(line.split(":", 1)[1].strip())
            elif line.startswith("temperature:"):
                current["temperature"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("avg_logprob:"):
                current["avg_logprob"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("compression_ratio:"):
                current["compression_ratio"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("no_speech_prob:"):
                current["no_speech_prob"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("-" * 5):  # Separator line
                if current:
                    segments.append(current)
                    current = {}

        # Add the last segment if file doesn't end with separator
        if current:
            segments.append(current)
    print("completed parsing......................")
    return segments


def merge_segments_into_sentences(segments):
    merged = []
    current = segments[0].copy()
    print("merging segments............................")
    for i in range(1, len(segments)):
        text = current["text"].strip()
        ends_sentence = re.search(r'[.!?]["\']?$|[\u06D4\u061F\u061B]$', text)  # Includes Arabic full stop/marks
        
        if ends_sentence:
            merged.append(current)
            current = segments[i].copy()
        else:
            # Merge with next segment
            next_seg = segments[i]
            current['id'].append(next_seg['id'][0])
            current["text"] += " " + next_seg["text"]
            current["end"] = next_seg["end"]
            current["tokens"].append(next_seg['tokens']) #next_seg.get("tokens", [])
            current["avg_logprob"] = (current["avg_logprob"] + next_seg["avg_logprob"]) / 2
            current["compression_ratio"] = (current["compression_ratio"] + next_seg["compression_ratio"]) / 2
            current["no_speech_prob"] = max(current["no_speech_prob"], next_seg["no_speech_prob"])

    # Add the last combined segment
    if current:
        merged.append(current)
    print("completed merging...........")
    return merged

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

def chunk_subtitle_segments(model_name, segments, similarity_threshold):
    print("Starting context-based chunking with metadata preservation...")
    print('initial length of segments',len(segments))
    # Step 1: Prepare sentences and map sentence indices to segments
    all_sentences = []
    sentence_to_segment = []
    segment_metadata = []

    for idx, seg in enumerate(segments):
        sentences = sent_tokenize(seg["text"].strip())
        for sent in sentences:
            all_sentences.append(sent)
            sentence_to_segment.append(idx)
        segment_metadata.append({
            "start": seg["start"],
            "end": seg["end"]
        })

    num_sentences = len(all_sentences)
    print(f"Total sentences: {num_sentences}")

    # Step 2: Encode sentences
    print("Loading model and encoding...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(all_sentences, normalize_embeddings=False)

    # Step 3: Similarity Matrix & TextRank Scores
    sim_matrix = cosine_similarity(embeddings)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    sentence_importance = {i: scores[i] for i in range(num_sentences)}

    # Step 4: Chunking based on similarity
    chunks = []
    current_chunk = [0]

    for i in range(1, num_sentences):
        last_index = current_chunk[-1]
        similarity = cosine_similarity([embeddings[last_index]], [embeddings[i]])[0][0]

        if similarity >= similarity_threshold:
            current_chunk.append(i)
        else:
            # Create chunk
            chunk_text = " ".join([all_sentences[j] for j in current_chunk])
            chunk_score = np.mean([sentence_importance[j] for j in current_chunk])
            involved_segments = [sentence_to_segment[j] for j in current_chunk]
            chunk_start = min([segment_metadata[k]["start"] for k in involved_segments])
            chunk_end = max([segment_metadata[k]["end"] for k in involved_segments])
            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_end,
                "score": chunk_score
            })
            current_chunk = [i]

    # Handle last chunk
    if current_chunk:
        chunk_text = " ".join([all_sentences[j] for j in current_chunk])
        chunk_score = np.mean([sentence_importance[j] for j in current_chunk])
        involved_segments = [sentence_to_segment[j] for j in current_chunk]
        chunk_start = min([segment_metadata[k]["start"] for k in involved_segments])
        chunk_end = max([segment_metadata[k]["end"] for k in involved_segments])
        chunks.append({
            "text": chunk_text,
            "start": chunk_start,
            "end": chunk_end,
            "score": chunk_score
        })

    print("Chunking complete.")
    return chunks

# def chunk_subtitle_segments(model_name, segments, similarity_threshold):
#     print("Starting improved context-based chunking...")

#     # Extract segment text and metadata
#     texts = [seg['text'].strip() for seg in segments]
#     starts = [seg['start'] for seg in segments]
#     ends = [seg['end'] for seg in segments]

#     print("Loading model...")
#     model = SentenceTransformer(model_name, trust_remote_code=True)
#     embeddings = model.encode(texts, normalize_embeddings=False)
#     print("Model loaded and text embedded.")

#     # Compute similarity matrix
#     sim_matrix = cosine_similarity(embeddings)

#     # Compute importance scores using TextRank
#     nx_graph = nx.from_numpy_array(sim_matrix)
#     scores = nx.pagerank(nx_graph)
#     sentence_importance = {i: scores[i] for i in range(len(texts))}

#     # Context-based chunking
#     chunks = []
#     current_chunk = [0]

#     for i in range(1, len(texts)):
#         last_index = current_chunk[-1]
#         similarity = cosine_similarity([embeddings[last_index]], [embeddings[i]])[0][0]

#         if similarity >= similarity_threshold:
#             current_chunk.append(i)
#         else:
#             chunk_text = " ".join([texts[j] for j in current_chunk])
#             chunk_start = starts[current_chunk[0]]
#             chunk_end = ends[current_chunk[-1]]
#             chunk_score = np.mean([sentence_importance[j] for j in current_chunk])

#             chunks.append({
#                 "text": chunk_text,
#                 "start": chunk_start,
#                 "end": chunk_end,
#                 "score": chunk_score
#             })
#             current_chunk = [i]

#     # Final chunk
#     if current_chunk:
#         chunk_text = " ".join([texts[j] for j in current_chunk])
#         chunk_start = starts[current_chunk[0]]
#         chunk_end = ends[current_chunk[-1]]
#         chunk_score = np.mean([sentence_importance[j] for j in current_chunk])

#         chunks.append({
#             "text": chunk_text,
#             "start": chunk_start,
#             "end": chunk_end,
#             "score": chunk_score
#         })

#     print("Chunking complete.")
#     return chunks


# def parse_segments_from_file(filepath):
#     segments = []
#     with open(filepath, 'r', encoding='utf-8') as file:
#         current = {}
#         for line in file:
#             line = line.strip()
#             if line.startswith("id:"):
#                 current = {"id": int(line.split(":")[1].strip())}
#             elif line.startswith("start:"):
#                 current["start"] = float(line.split(":")[1].strip())
#             elif line.startswith("end:"):
#                 current["end"] = float(line.split(":")[1].strip())
#             elif line.startswith("text:"):
#                 current["text"] = line.split(":", 1)[1].strip()
#             elif line.startswith("----------------------------------------------------------------------------------"):
#                 if current:
#                     segments.append(current)
#                     current = {}
#         # Append last block if file does not end with separator
#         if current:
#             segments.append(current)
#     return segments


filepath='video_samples/xjGJ5wYs8AQ_10minutes/formatted_transcript.txt'
segments = parse_segments_from_file(filepath)
model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
threshold_similarity=0.3
merged_segments = merge_segments_into_sentences(segments)

# # Step 2: Get context-based chunks with scores and timing
chunks= chunk_subtitle_segments(model_name,merged_segments,threshold_similarity)

for i,c in enumerate(chunks,1):
    print(f'chunk number: {i}')
    print(c)
# # Optional: Print or save output
# for i, ch in enumerate(chunks, 1):
#     print(f"Chunk {i} | {ch['start']} - {ch['end']} | Score: {ch['score']:.4f}\n{ch['text']}\n")
