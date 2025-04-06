import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def context_based_chunking(model_name,text, similarity_threshold):
    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    print(f"Total sentences: {num_sentences}")
    print("loading model................")
    # Step 2: Sentence Embedding using SBERT
    print(model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    print("model loaded................")
    print("encoding sentences............")
    embeddings = model.encode(sentences,normalize_embeddings=False)
    print(embeddings.shape)
    print("encoded sentences .........")
    # Step 3: Similarity Matrix
    print("genrating similarity matrix........................")
    sim_matrix = cosine_similarity(embeddings)
    print("generated similarity matrix........................")

    # Step 4: Sentence Importance via TextRank
    print(" Step 4: Sentence Importance via TextRank")
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    sentence_importance = {i: scores[i] for i in range(num_sentences)}

    # Step 5: Context-based Chunking
    chunks = []
    chunk_scores = []

    current_chunk = [0]
    print("starting loop for      Step 5: Context-based Chunking")
    for i in range(1, num_sentences):
        last_index = current_chunk[-1]
        similarity = cosine_similarity([embeddings[last_index]], [embeddings[i]])[0][0]

        if similarity >= similarity_threshold:
            current_chunk.append(i)
        else:
            chunk_text = " ".join([sentences[j] for j in current_chunk])
            avg_score = np.mean([sentence_importance[j] for j in current_chunk])
            chunks.append(chunk_text)
            chunk_scores.append(avg_score)
            current_chunk = [i]

    # Handle last chunk
    if current_chunk:
        chunk_text = " ".join([sentences[j] for j in current_chunk])
        avg_score = np.mean([sentence_importance[j] for j in current_chunk])
        chunks.append(chunk_text)
        chunk_scores.append(avg_score)

    # Step 6: Combine chunks with scores
    
    chunk_data = [{"chunk": chunk, "score": score} for chunk, score in zip(chunks, chunk_scores)]
    print("ended chunking ..................")
    return chunk_data

filepath='video_samples/6vX3Us1TOw8_14minutes/full_transcript.txt'
text=(open(filepath,'r',encoding='utf-8')).read()
model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

threshold_similarity=0.5
chunks_with_scores = context_based_chunking(model_name,text,threshold_similarity)

output_file = f'video_samples/6vX3Us1TOw8_14minutes/outputs/{(model_name.split('/'))[1]}_threshold{threshold_similarity}_chunking.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for i, data in enumerate(chunks_with_scores, 1):
        file.write(f"Chunk {i} (Score: {data['score']:.4f}):\n{data['chunk']}\n\n")

print(f"Output with importance scores has been written to {output_file}")