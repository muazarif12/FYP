

import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def textrank_segmentation_with_scores(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_chunks=num_sentences//5 # a cluster or chunk cannot have more than 5 sentences
    print("number of sentences= ", num_sentences)

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    # Compute sentence similarity matrix
    sim_matrix = cosine_similarity(X)

    # Apply TextRank (graph-based ranking)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Store sentence importance
    sentence_importance = {i: scores[i] for i in range(num_sentences)}

    # Determine chunking strategy
    chunk_size = num_sentences // num_chunks
    remainder = num_sentences % num_chunks

    chunks = []
    chunk_scores = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_sentences = sentences[start:end]
        chunk = " ".join(chunk_sentences)
        score = np.mean([sentence_importance[j] for j in range(start, end)])
        chunks.append(chunk)
        chunk_scores.append(score)

    # Add remaining sentences to the last chunk
    if remainder > 0:
        start = num_chunks* chunk_size
        remaining_sentences = sentences[start:]
        remaining_chunk = " ".join(remaining_sentences)
        score = np.mean([sentence_importance[j] for j in range(start, num_sentences)])
        chunks.append(remaining_chunk)
        chunk_scores.append(score)

    # Combine chunks and scores
    chunk_data = [{"chunk": chunk, "score": score} for chunk, score in zip(chunks, chunk_scores)]
    return chunk_data
filepath='video_samples/xjGJ5wYs8AQ_10minutes/full_transcript.txt'
text=(open(filepath,'r',encoding='utf-8')).read()
chunks_with_scores = textrank_segmentation_with_scores(text)

output_file = 'output_textrank_segmentation_with_scores_10minutes.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for i, data in enumerate(chunks_with_scores, 1):
        file.write(f"Chunk {i} (Score: {data['score']:.4f}):\n{data['chunk']}\n\n")

print(f"Output with importance scores has been written to {output_file}")