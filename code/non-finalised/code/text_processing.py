from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from bertopic import BERTopic

def chunk_subtitle_segments(model_name, sentences_data, similarity_threshold):
    """
    Create chunks of similar sentences based on semantic similarity.
    
    Args:
        model_name: Name of the sentence transformer model to use
        sentences_data: List of dictionaries with 'id', 'sentence', 'start', 'end' keys
        similarity_threshold: Threshold value for similarity to consider sentences as part of the same chunk
        
    Returns:
        List of chunks with text, start time, end time, and importance score
    """
    print("Starting context-based chunking with metadata preservation...")
    print(f'Initial length of sentences: {len(sentences_data)}')
    
    # Step 1: Extract sentences
    all_sentences = [item['sentence'] for item in sentences_data]

    num_sentences = len(all_sentences)
    print(f"Total sentences: {num_sentences}")
    
    # Step 2: Encode sentences
    print("Loading model and encoding...")
    embeddings=get_embeddings(model_name,all_sentences)
    # Step 3: Calculate sentence importance using TextRank
    # sentence_importance = get_importance_textRank(embeddings,sentences_data)
    
    # Step 4: Create chunks based on similarity
    chunks = []
    current_chunk_indices = [0]
    current_chunk_sentences = [all_sentences[0]]
    
    for i in range(1, num_sentences):
        last_index = current_chunk_indices[-1]
        similarity = cosine_similarity([embeddings[last_index]], [embeddings[i]])[0][0]
        
        if similarity >= similarity_threshold:
            # Add to current chunk
            current_chunk_indices.append(i)
            current_chunk_sentences.append(all_sentences[i])
        else:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            # chunk_score = np.mean([sentence_importance[j] for j in current_chunk_indices])
            
            # Get start time from first sentence and end time from last sentence in the chunk
            chunk_start = sentences_data[current_chunk_indices[0]]["start"]
            chunk_end = sentences_data[current_chunk_indices[-1]]["end"]
            
            chunks.append({
                "sentence": chunk_text,
                "start": chunk_start,
                "end": chunk_end,
                # "score": chunk_score
            })
            
            # Start a new chunk
            current_chunk_indices = [i]
            current_chunk_sentences = [all_sentences[i]]
    
    # Handle the last chunk
    if current_chunk_indices:
        chunk_text = " ".join(current_chunk_sentences)
        # chunk_score = np.mean([sentence_importance[j] for j in current_chunk_indices])
        
        chunk_start = sentences_data[current_chunk_indices[0]]["start"]
        chunk_end = sentences_data[current_chunk_indices[-1]]["end"]
        
        chunks.append({
            "sentence": chunk_text,
            "start": chunk_start,
            "end": chunk_end,
            # "score": chunk_score
        })
    
    print(f"Chunking complete. Created {len(chunks)} chunks.")
    return chunks


def sentences_with_topic_modelling(sentences_data,vectoriser=None,model=None):
    all_sentences = [item['sentence'] for item in sentences_data]
    if model is not None and vectoriser is not None:
        print('model: ',model,' vectoriser: ',vectoriser)
        topic_model = BERTopic(embedding_model=model,vectorizer_model=vectoriser)
    elif vectoriser is not None:
        print('model: ',model,' vectoriser: ',vectoriser)
        topic_model = BERTopic(vectorizer_model=vectoriser)
    elif model is not None:
        print('model: ',model,' vectoriser: ',vectoriser)
        topic_model=BERTopic(embedding_model=model)
    else:
        print('model: ',model,' vectoriser: ',vectoriser)
        topic_model=BERTopic()
    topics, _ = topic_model.fit_transform(all_sentences)
    for i, item in enumerate(sentences_data):
        item['topic']=topics[i]
    return sentences_data

def sentiment_analyser(chunks):
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt)
    for chunk in chunks:
        sentence = chunk.get('text', '')
        if sentence:
            # Run sentiment analysis on the sentence
            result = pipe(sentence,top_k=1,truncation=True)[0]  # result is a list with one dictionary
            # Add sentiment score to the original dictionary
            chunk['sentiment_score'] = result['score']
            chunk['sentiment_label'] = result['label']

    return chunks

def get_importance_textRank(embeddings,sentences_data):
    all_sentences=[item['sentence'] for item in sentences_data]
    sim_matrix = cosine_similarity(embeddings)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    sentence_importance = {i: scores[i] for i in range(len(all_sentences))}
    return sentence_importance

def get_embeddings(model,all_sentences):
    model = SentenceTransformer(model, trust_remote_code=True)
    embeddings = model.encode(all_sentences, normalize_embeddings=False)
    return embeddings

def product_importance_sentiments(chunks):
    pass
    
# def generate_extractive_summary_1(
#     model_name,
#     sentence_list,
#     language="english",
#     summary_ratio=0.3,
#     min_duration=5.0,
#     min_topic_size=2,
#     similarity_threshold_english=0.3,
#     similarity_threshold_arabic=0.25,
#     pagerank_alpha=0.85,
#     mmr_lambda_english=0.6,
#     mmr_lambda_arabic=0.7
# ):
#     """
#     Generate extractive summary from a list of sentence dictionaries.

#     Args:
#         model_name: SentenceTransformer model name for embedding generation.
#         sentence_list: List of dictionaries with keys: 'id', 'sentence', 'start', 'end'
#         language: Language of text, 'english' or 'arabic'
#         summary_ratio: Percentage of sentences to include in summary (per topic)
#         min_duration: Minimum duration (in seconds) to keep a subtitle segment
#         min_topic_size: Minimum number of sentences in a topic for BERTopic
#         similarity_threshold_english: Cosine similarity threshold for English
#         similarity_threshold_arabic: Cosine similarity threshold for Arabic
#         pagerank_alpha: Damping factor for PageRank algorithm
#         mmr_lambda_english: Lambda for MMR (balance between relevance/diversity) for English
#         mmr_lambda_arabic: Lambda for MMR for Arabic

#     Returns:
#         List of dictionaries representing the summary sentences
#     """
#     # 1. Extract sentences and durations
#     sentences = [item['sentence'] for item in sentence_list]
#     print('printing chunks from claude suggested')
#     for i in sentences:
#         print(i)
#     durations = [item['end'] - item['start'] for item in sentence_list]
    
#     # 2. Filter by duration
#     filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
#     if not filtered_indices:
#         filtered_indices = list(range(len(sentences)))
    
#     filtered_sentences = [sentences[i] for i in filtered_indices]
    
#     # 3. Generate embeddings
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(filtered_sentences)
    
#     # 4. Topic modeling
#     from bertopic import BERTopic
#     topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size,embedding_model=model_name)
    
#     if len(filtered_sentences) < 4:
#         topics = [0] * len(filtered_sentences)
#     else:
#         try:
#             topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
#         except:
#             topics = [0] * len(filtered_sentences)
    
#     # 5. Similarity matrix
#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity

#     threshold = similarity_threshold_english if language == "english" else similarity_threshold_arabic
#     similarity_matrix = cosine_similarity(embeddings)
#     similarity_matrix[similarity_matrix < threshold] = 0
    
#     # 6. Group sentences by topic
#     topic_groups = {}
#     for i, topic in enumerate(topics):
#         if topic not in topic_groups:
#             topic_groups[topic] = []
#         topic_groups[topic].append(i)
    
#     # 7. PageRank and MMR
#     import networkx as nx

#     selected_indices = []
#     topic_scores = {}

#     for topic, indices in topic_groups.items():
#         if not indices:
#             continue
        
#         if len(indices) == 1:
#             selected_indices.append(filtered_indices[indices[0]])
#             topic_scores[filtered_indices[indices[0]]] = 1.0
#             continue
        
#         topic_sim_matrix = np.zeros((len(indices), len(indices)))
#         for i, idx1 in enumerate(indices):
#             for j, idx2 in enumerate(indices):
#                 topic_sim_matrix[i, j] = similarity_matrix[idx1, idx2]
        
#         nx_graph = nx.from_numpy_array(topic_sim_matrix)
#         try:
#             scores = nx.pagerank(nx_graph, alpha=pagerank_alpha)

#             for i, idx in enumerate(indices):
#                 original_idx = filtered_indices[idx]
#                 duration = durations[original_idx]
#                 scores[i] = scores[i] * np.log1p(duration)
#                 topic_scores[original_idx] = scores[i]
            
#             lambda_val = mmr_lambda_english if language == "english" else mmr_lambda_arabic
#             n_to_select = max(1, int(len(indices) * summary_ratio))
            
#             selected = []
#             unselected = list(range(len(indices)))
            
#             first_idx = max(unselected, key=lambda i: scores[i])
#             selected.append(first_idx)
#             unselected.remove(first_idx)

#             while len(selected) < n_to_select and unselected:
#                 mmr_scores = []
#                 for i in unselected:
#                     relevance = scores[i]
#                     diversity = max([topic_sim_matrix[i, j] for j in selected]) if selected else 0
#                     mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
#                     mmr_scores.append((i, mmr_score))
                
#                 next_idx, _ = max(mmr_scores, key=lambda x: x[1])
#                 selected.append(next_idx)
#                 unselected.remove(next_idx)
            
#             selected_indices.extend([filtered_indices[indices[i]] for i in selected])

#         except:
#             n_to_select = max(1, int(len(indices) * summary_ratio))
#             selected = indices[:n_to_select]
#             selected_indices.extend([filtered_indices[i] for i in selected])
#             for i in selected:
#                 topic_scores[filtered_indices[i]] = 1.0
    
#     # 8. Sort to maintain order
#     selected_indices.sort()

#     # 9. Prepare final summary
#     summary = []
#     for i, idx in enumerate(selected_indices):
#         original_item = sentence_list[idx]
#         summary_item = {
#             'id': i + 1,
#             'sentence': original_item['sentence'],
#             'start': original_item['start'],
#             'end': original_item['end'],
#             'topic': topics[filtered_indices.index(idx)] if idx in filtered_indices else -1,
#             'score': topic_scores.get(idx, 0)
#         }
#         summary.append(summary_item)
    
#     return summary

def generate_extractive_summary_1(
    model_name,
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold_english=0.3,
    similarity_threshold_arabic=0.25,
    pagerank_alpha=0.85,
    mmr_lambda_english=0.6,
    mmr_lambda_arabic=0.7,
    filter_negative_topics=True  # Parameter to control topic filtering
):
    """
    Generate extractive summary from a list of sentence dictionaries.

    Args:
        model_name: SentenceTransformer model name for embedding generation.
        sentence_list: List of dictionaries with keys: 'id', 'sentence', 'start', 'end'
        language: Language of text, 'english' or 'arabic'
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold_english: Cosine similarity threshold for English
        similarity_threshold_arabic: Cosine similarity threshold for Arabic
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda_english: Lambda for MMR (balance between relevance/diversity) for English
        mmr_lambda_arabic: Lambda for MMR for Arabic
        filter_negative_topics: Whether to remove sentences with topic value of -1

    Returns:
        List of dictionaries representing the summary sentences
    """
    # 1. Extract sentences and durations
    sentences = [item['sentence'] for item in sentence_list]
    print('printing chunks from claude suggested')
    for i in sentences:
        print(i)
    durations = [item['end'] - item['start'] for item in sentence_list]
    
    # 2. Filter by duration
    filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
    if not filtered_indices:
        filtered_indices = list(range(len(sentences)))
    
    filtered_sentences = [sentences[i] for i in filtered_indices]
    
    # 3. Generate embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(filtered_sentences)
    
    # 4. Topic modeling
    from bertopic import BERTopic
    topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size,embedding_model=model_name)
    
    if len(filtered_sentences) < 4:
        topics = [0] * len(filtered_sentences)
    else:
        try:
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except:
            topics = [0] * len(filtered_sentences)
    
    # 5. Filter out sentences with topic -1 right after topic modeling
    if filter_negative_topics:
        valid_indices = []
        valid_sentences = []
        valid_topics = []
        valid_embeddings = []
        
        for i, (idx, topic) in enumerate(zip(filtered_indices, topics)):
            if topic != -1:
                valid_indices.append(idx)
                valid_sentences.append(filtered_sentences[i])
                valid_topics.append(topic)
                valid_embeddings.append(embeddings[i])
        
        # Replace the original arrays with filtered versions
        filtered_indices = valid_indices
        filtered_sentences = valid_sentences
        topics = valid_topics
        embeddings = valid_embeddings
        
        if len(filtered_indices) == 0:
            print("Warning: All sentences were filtered out due to having topic -1")
            return []
    
    # 6. Similarity matrix
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    threshold = similarity_threshold_english if language == "english" else similarity_threshold_arabic
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix[similarity_matrix < threshold] = 0
    
    # 7. Group sentences by topic
    topic_groups = {}
    for i, topic in enumerate(topics):
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # 8. PageRank and MMR
    import networkx as nx

    selected_indices = []
    topic_scores = {}

    for topic, indices in topic_groups.items():
        if not indices:
            continue
        
        if len(indices) == 1:
            selected_indices.append(filtered_indices[indices[0]])
            topic_scores[filtered_indices[indices[0]]] = 1.0
            continue
        
        topic_sim_matrix = np.zeros((len(indices), len(indices)))
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                topic_sim_matrix[i, j] = similarity_matrix[idx1, idx2]
        
        nx_graph = nx.from_numpy_array(topic_sim_matrix)
        try:
            scores = nx.pagerank(nx_graph, alpha=pagerank_alpha)

            for i, idx in enumerate(indices):
                original_idx = filtered_indices[idx]
                duration = durations[original_idx]
                scores[i] = scores[i] * np.log1p(duration)
                topic_scores[original_idx] = scores[i]
            
            lambda_val = mmr_lambda_english if language == "english" else mmr_lambda_arabic
            n_to_select = max(1, int(len(indices) * summary_ratio))
            
            selected = []
            unselected = list(range(len(indices)))
            
            first_idx = max(unselected, key=lambda i: scores[i])
            selected.append(first_idx)
            unselected.remove(first_idx)

            while len(selected) < n_to_select and unselected:
                mmr_scores = []
                for i in unselected:
                    relevance = scores[i]
                    diversity = max([topic_sim_matrix[i, j] for j in selected]) if selected else 0
                    mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([filtered_indices[indices[i]] for i in selected])

        except:
            n_to_select = max(1, int(len(indices) * summary_ratio))
            selected = indices[:n_to_select]
            selected_indices.extend([filtered_indices[i] for i in selected])
            for i in selected:
                topic_scores[filtered_indices[i]] = 1.0
    
    # 9. Sort to maintain order
    selected_indices.sort()

    # 10. Prepare final summary
    summary = []
    topic_map = {filtered_indices[i]: topics[i] for i in range(len(filtered_indices))}
    
    for i, idx in enumerate(selected_indices):
        original_item = sentence_list[idx]
        
        summary_item = {
            'id': i + 1,
            'sentence': original_item['sentence'],
            'start': original_item['start'],
            'end': original_item['end'],
            'topic': topic_map.get(idx, 0),  # Use 0 as fallback instead of -1
            'score': topic_scores.get(idx, 0)
        }
        summary.append(summary_item)
    
    return summary


def topic_based_chunking_and_summarization(
    model_name,
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold_english=0.3,
    similarity_threshold_arabic=0.25,
    pagerank_alpha=0.85,
    mmr_lambda_english=0.6,
    mmr_lambda_arabic=0.7
):
    """
    Generate extractive summary using topic-based chunking approach.
    
    Args:
        model_name: SentenceTransformer model name for embedding generation.
        sentence_list: List of dictionaries with keys: 'id', 'sentence', 'start', 'end'
        language: Language of text, 'english' or 'arabic'
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold_english: Cosine similarity threshold for English
        similarity_threshold_arabic: Cosine similarity threshold for Arabic
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda_english: Lambda for MMR (balance between relevance/diversity) for English
        mmr_lambda_arabic: Lambda for MMR for Arabic
        
    Returns:
        List of dictionaries representing the summary sentences
    """
    # 1. Extract sentences and durations
    sentences = [item['sentence'] for item in sentence_list]
    durations = [item['end'] - item['start'] for item in sentence_list]
    
    # 2. Filter by duration
    filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
    if not filtered_indices:
        filtered_indices = list(range(len(sentences)))
    
    filtered_sentences = [sentences[i] for i in filtered_indices]
    filtered_items = [sentence_list[i] for i in filtered_indices]
    
    # 3. Generate embeddings

    embeddings = get_embeddings(model_name,filtered_sentences)
    
    # 4. Topic modeling first (before chunking)
    from bertopic import BERTopic
    topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size,embedding_model=model_name)
    
    if len(filtered_sentences) < 4:
        topics = [0] * len(filtered_sentences)
    else:
        try:
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except:
            topics = [0] * len(filtered_sentences)
    
    # 5. Add topic information to the filtered items
    for i, topic in enumerate(topics):
        filtered_items[i]['topic'] = topic
    
    # 6. Similarity matrix calculation
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    threshold = similarity_threshold_english if language == "english" else similarity_threshold_arabic
    similarity_matrix = cosine_similarity(embeddings)
    
    # 7. Chunk sentences based on similarity AND same topic
    chunked_items = []
    
    # Start with the first sentence
    current_chunk = [filtered_items[0]]
    current_topic = filtered_items[0]['topic']
    
    for i in range(1, len(filtered_items)):
        # Check if current sentence is similar to the last sentence in the chunk
        last_index = filtered_indices.index(filtered_indices[filtered_items.index(current_chunk[-1])])
        current_index = filtered_indices.index(filtered_indices[i])
        
        similarity = similarity_matrix[last_index, current_index]
        same_topic = filtered_items[i]['topic'] == current_topic
        
        # If similar and same topic, add to current chunk
        if similarity >= threshold and same_topic:
            current_chunk.append(filtered_items[i])
        else:
            # Process the completed chunk
            if current_chunk:
                # Create a single chunk item that combines all sentences
                chunk_text = " ".join([item['sentence'] for item in current_chunk])
                chunk_start = current_chunk[0]['start']
                chunk_end = current_chunk[-1]['end']
                
                chunk_item = {
                    'sentence': chunk_text,
                    'start': chunk_start,
                    'end': chunk_end,
                    'topic': current_topic
                }
                chunked_items.append(chunk_item)
            
            # Start a new chunk
            current_chunk = [filtered_items[i]]
            current_topic = filtered_items[i]['topic']
    
    # Process the last chunk
    if current_chunk:
        chunk_text = " ".join([item['sentence'] for item in current_chunk])
        chunk_start = current_chunk[0]['start']
        chunk_end = current_chunk[-1]['end']
        
        chunk_item = {
            'sentence': chunk_text,
            'start': chunk_start,
            'end': chunk_end,
            'topic': current_topic
        }
        chunked_items.append(chunk_item)
    
    print(f"Created {len(chunked_items)} topic-based chunks from {len(filtered_items)} sentences")

    # 8. Group chunks by topic
    topic_groups = {}
    for i, item in enumerate(chunked_items):
        topic = item['topic']
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # ----- The rest of the function follows the original extractive summary approach -----
    
    # 9. Re-encode the chunk sentences for analysis
    chunk_sentences = [item['sentence'] for item in chunked_items]
    chunk_embeddings = get_embeddings(model_name,chunk_sentences)
    
    # 10. Calculate new similarity matrix for chunks
    chunk_similarity_matrix = cosine_similarity(chunk_embeddings)
    chunk_similarity_matrix[chunk_similarity_matrix < threshold] = 0
    
    # 11. PageRank and MMR for selection
    import networkx as nx
    
    selected_indices = []
    topic_scores = {}
    
    for topic, indices in topic_groups.items():
        if not indices:
            continue
        
        if len(indices) == 1:
            selected_indices.append(indices[0])
            topic_scores[indices[0]] = 1.0
            continue
        
        topic_sim_matrix = np.zeros((len(indices), len(indices)))
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                topic_sim_matrix[i, j] = chunk_similarity_matrix[idx1, idx2]
        
        nx_graph = nx.from_numpy_array(topic_sim_matrix)
        try:
            scores = nx.pagerank(nx_graph, alpha=pagerank_alpha)
            
            # Apply duration boost
            for i, idx in enumerate(indices):
                chunk_duration = chunked_items[idx]['end'] - chunked_items[idx]['start']
                scores[i] = scores[i] * np.log1p(chunk_duration)
                topic_scores[idx] = scores[i]
            
            # Apply MMR to select diverse yet important chunks
            lambda_val = mmr_lambda_english if language == "english" else mmr_lambda_arabic
            n_to_select = max(1, int(len(indices) * summary_ratio))
            
            selected = []
            unselected = list(range(len(indices)))
            
            # Select first by highest score
            first_idx = max(unselected, key=lambda i: scores[i])
            selected.append(first_idx)
            unselected.remove(first_idx)
            
            # Select the rest using MMR
            while len(selected) < n_to_select and unselected:
                mmr_scores = []
                for i in unselected:
                    relevance = scores[i]
                    diversity = max([topic_sim_matrix[i, j] for j in selected]) if selected else 0
                    mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([indices[i] for i in selected])
            
        except:
            # Fallback if PageRank fails
            n_to_select = max(1, int(len(indices) * summary_ratio))
            selected = indices[:n_to_select]
            selected_indices.extend(selected)
            for i in selected:
                topic_scores[i] = 1.0
    
    # 12. Sort to maintain chronological order
    selected_indices.sort()
    
    # 13. Prepare final summary
    summary = []
    for i, idx in enumerate(selected_indices):
        chunk_item = chunked_items[idx]
        summary_item = {
            'id': i + 1,
            'sentence': chunk_item['sentence'],
            'start': chunk_item['start'],
            'end': chunk_item['end'],
            'topic': chunk_item['topic'],
            'score': topic_scores.get(idx, 0)
        }
        summary.append(summary_item)
    
    return summary


def topic_based_chunking_and_summarization_with_sentiments(
    model_name,
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold_english=0.3,
    similarity_threshold_arabic=0.25,
    pagerank_alpha=0.85,
    mmr_lambda_english=0.6,
    mmr_lambda_arabic=0.7
):
    """
    Generate extractive summary using topic-based chunking approach with sentiment analysis.
    
    Args:
        model_name: SentenceTransformer model name for embedding generation.
        sentence_list: List of dictionaries with keys: 'id', 'sentence', 'start', 'end'
        language: Language of text, 'english' or 'arabic'
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold_english: Cosine similarity threshold for English
        similarity_threshold_arabic: Cosine similarity threshold for Arabic
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda_english: Lambda for MMR (balance between relevance/diversity) for English
        mmr_lambda_arabic: Lambda for MMR for Arabic
        
    Returns:
        List of dictionaries representing the summary sentences
    """
    # 1. Extract sentences and durations
    sentences = [item['sentence'] for item in sentence_list]
    durations = [item['end'] - item['start'] for item in sentence_list]
    
    # 2. Filter by duration
    filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
    if not filtered_indices:
        filtered_indices = list(range(len(sentences)))
    
    filtered_sentences = [sentences[i] for i in filtered_indices]
    filtered_items = [sentence_list[i] for i in filtered_indices]
    
    # 3. Generate embeddings
    embeddings = get_embeddings(model_name, filtered_sentences)
    
    # 4. Topic modeling first (before chunking)
    from bertopic import BERTopic
    topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size,embedding_model=model_name)
    
    if len(filtered_sentences) < 4:
        topics = [0] * len(filtered_sentences)
    else:
        try:
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except:
            topics = [0] * len(filtered_sentences)
    
    # 5. Add topic information to the filtered items
    for i, topic in enumerate(topics):
        filtered_items[i]['topic'] = topic
    
    # 6. Similarity matrix calculation
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    threshold = similarity_threshold_english if language == "english" else similarity_threshold_arabic
    similarity_matrix = cosine_similarity(embeddings)
    
    # 7. Chunk sentences based on similarity AND same topic
    chunked_items = []
    
    # Start with the first sentence
    current_chunk = [filtered_items[0]]
    current_topic = filtered_items[0]['topic']
    
    for i in range(1, len(filtered_items)):
        # Check if current sentence is similar to the last sentence in the chunk
        last_index = filtered_indices.index(filtered_indices[filtered_items.index(current_chunk[-1])])
        current_index = filtered_indices.index(filtered_indices[i])
        
        similarity = similarity_matrix[last_index, current_index]
        same_topic = filtered_items[i]['topic'] == current_topic
        
        # If similar and same topic, add to current chunk
        if similarity >= threshold and same_topic:
            current_chunk.append(filtered_items[i])
        else:
            # Process the completed chunk
            if current_chunk:
                # Create a single chunk item that combines all sentences
                chunk_text = " ".join([item['sentence'] for item in current_chunk])
                chunk_start = current_chunk[0]['start']
                chunk_end = current_chunk[-1]['end']
                
                chunk_item = {
                    'sentence': chunk_text,
                    'start': chunk_start,
                    'end': chunk_end,
                    'topic': current_topic
                }
                chunked_items.append(chunk_item)
            
            # Start a new chunk
            current_chunk = [filtered_items[i]]
            current_topic = filtered_items[i]['topic']
    
    # Process the last chunk
    if current_chunk:
        chunk_text = " ".join([item['sentence'] for item in current_chunk])
        chunk_start = current_chunk[0]['start']
        chunk_end = current_chunk[-1]['end']
        
        chunk_item = {
            'sentence': chunk_text,
            'start': chunk_start,
            'end': chunk_end,
            'topic': current_topic
        }
        chunked_items.append(chunk_item)
    
    print(f"Created {len(chunked_items)} topic-based chunks from {len(filtered_items)} sentences")

    # 8. Group chunks by topic
    topic_groups = {}
    for i, item in enumerate(chunked_items):
        topic = item['topic']
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # 9. Re-encode the chunk sentences for analysis
    chunk_sentences = [item['sentence'] for item in chunked_items]
    chunk_embeddings = get_embeddings(model_name, chunk_sentences)
    
    # 10. Calculate new similarity matrix for chunks
    chunk_similarity_matrix = cosine_similarity(chunk_embeddings)
    chunk_similarity_matrix[chunk_similarity_matrix < threshold] = 0
    
    # 11. PageRank and MMR for selection
    import networkx as nx
    
    selected_indices = []
    topic_scores = {}
    
    # 12. Sentiment Analysis integration
    # Prepare chunks for sentiment analysis
    chunks_for_sentiment = [{'text': item['sentence']} for item in chunked_items]
    
    # Run sentiment analysis on all chunks
    chunks_with_sentiment = sentiment_analyser(chunks_for_sentiment)
    
    # Extract sentiment scores and normalize them to range 0-1
    sentiment_scores = {}
    for i, chunk in enumerate(chunks_with_sentiment):
        # Map sentiment labels to numerical values (assuming POSITIVE=1, NEUTRAL=0, NEGATIVE=-1)
        label = chunk.get('sentiment_label', 'NEUTRAL')
        label_value = 1 if label == 'POSITIVE' else (-1 if label == 'NEGATIVE' else 0)
        
        # Normalize to 0-1 range (from -1 to 1)
        normalized_sentiment = (label_value + 1) / 2
        
        # Store the sentiment score
        sentiment_scores[i] = normalized_sentiment * chunk.get('sentiment_score', 1.0)
    
    for topic, indices in topic_groups.items():
        if not indices:
            continue
        
        if len(indices) == 1:
            # Apply sentiment score directly for single chunks
            final_score = sentiment_scores.get(indices[0], 1.0)
            topic_scores[indices[0]] = final_score
            selected_indices.append(indices[0])
            continue
        
        topic_sim_matrix = np.zeros((len(indices), len(indices)))
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                topic_sim_matrix[i, j] = chunk_similarity_matrix[idx1, idx2]
        
        nx_graph = nx.from_numpy_array(topic_sim_matrix)
        try:
            scores = nx.pagerank(nx_graph, alpha=pagerank_alpha)
            
            # Apply duration boost and sentiment score
            for i, idx in enumerate(indices):
                chunk_duration = chunked_items[idx]['end'] - chunked_items[idx]['start']
                
                # Multiply PageRank score by log duration and sentiment score
                pagerank_score = scores[i] * np.log1p(chunk_duration)
                sentiment_factor = sentiment_scores.get(idx, 1.0)
                
                # Calculate final score as product of PageRank and sentiment
                final_score = pagerank_score * sentiment_factor
                topic_scores[idx] = final_score
            
            # Apply MMR to select diverse yet important chunks
            lambda_val = mmr_lambda_english if language == "english" else mmr_lambda_arabic
            n_to_select = max(1, int(len(indices) * summary_ratio))
            
            selected = []
            unselected = list(range(len(indices)))
            
            # Select first by highest score (now incorporating sentiment)
            first_idx = max(unselected, key=lambda i: topic_scores[indices[i]])
            selected.append(first_idx)
            unselected.remove(first_idx)
            
            # Select the rest using MMR
            while len(selected) < n_to_select and unselected:
                mmr_scores = []
                for i in unselected:
                    relevance = topic_scores[indices[i]]
                    diversity = max([topic_sim_matrix[i, j] for j in selected]) if selected else 0
                    mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([indices[i] for i in selected])
            
        except:
            # Fallback if PageRank fails
            n_to_select = max(1, int(len(indices) * summary_ratio))
            # Sort indices by sentiment scores
            sorted_indices = sorted(indices, key=lambda idx: sentiment_scores.get(idx, 0.0), reverse=True)
            selected = sorted_indices[:n_to_select]
            selected_indices.extend(selected)
            for i in selected:
                topic_scores[i] = sentiment_scores.get(i, 1.0)
    
    # 13. Sort to maintain chronological order
    selected_indices.sort()
    
    # 14. Prepare final summary
    summary = []
    for i, idx in enumerate(selected_indices):
        chunk_item = chunked_items[idx]
        summary_item = {
            'id': i + 1,
            'sentence': chunk_item['sentence'],
            'start': chunk_item['start'],
            'end': chunk_item['end'],
            'topic': chunk_item['topic'],
            'score': topic_scores.get(idx, 0),
            'sentiment_label': chunks_with_sentiment[idx].get('sentiment_label', 'NEUTRAL'),
            'sentiment_score': chunks_with_sentiment[idx].get('sentiment_score', 0)
        }
        summary.append(summary_item)
    
    return summary


def generate_extractive_summary_2(
    model_name,
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold_english=0.3,
    similarity_threshold_arabic=0.25,
    pagerank_alpha=0.85,
    mmr_lambda_english=0.6,
    mmr_lambda_arabic=0.7,
    filter_negative_topics=True  # New parameter to control topic filtering
):
    """
    Generate extractive summary from a list of sentence dictionaries with sentiment analysis integration.

    Args:
        model_name: SentenceTransformer model name for embedding generation.
        sentence_list: List of dictionaries with keys: 'id', 'sentence', 'start', 'end'
        language: Language of text, 'english' or 'arabic'
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold_english: Cosine similarity threshold for English
        similarity_threshold_arabic: Cosine similarity threshold for Arabic
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda_english: Lambda for MMR (balance between relevance/diversity) for English
        mmr_lambda_arabic: Lambda for MMR for Arabic
        filter_negative_topics: Whether to remove sentences with topic value of -1

    Returns:
        List of dictionaries representing the summary sentences with sentiment information
    """
    # 1. Extract sentences and durations
    sentences = [item['sentence'] for item in sentence_list]
    print('printing chunks from claude suggested')
    for i in sentences:
        print(i)
    durations = [item['end'] - item['start'] for item in sentence_list]
    
    # 2. Filter by duration
    filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
    if not filtered_indices:
        filtered_indices = list(range(len(sentences)))
    
    filtered_sentences = [sentences[i] for i in filtered_indices]
    
    # 3. Generate embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(filtered_sentences)
    
    # 4. Topic modeling
    from bertopic import BERTopic
    topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size, embedding_model=model_name)
    
    if len(filtered_sentences) < 4:
        topics = [0] * len(filtered_sentences)
    else:
        try:
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except:
            topics = [0] * len(filtered_sentences)
    
    # 5. Filter out sentences with topic -1 right after topic modeling
    if filter_negative_topics:
        valid_indices = []
        valid_sentences = []
        valid_topics = []
        valid_embeddings = []
        
        for i, (idx, topic) in enumerate(zip(filtered_indices, topics)):
            if topic != -1:
                valid_indices.append(idx)
                valid_sentences.append(filtered_sentences[i])
                valid_topics.append(topic)
                valid_embeddings.append(embeddings[i])
        
        # Replace the original arrays with filtered versions
        filtered_indices = valid_indices
        filtered_sentences = valid_sentences
        topics = valid_topics
        embeddings = valid_embeddings
        
        if len(filtered_indices) == 0:
            print("Warning: All sentences were filtered out due to having topic -1")
            return []
    
    # 6. Perform sentiment analysis on filtered sentences
    sentences_for_sentiment = [{'text': sentence} for sentence in filtered_sentences]
    sentences_with_sentiment = sentiment_analyser(sentences_for_sentiment)
    
    # 7. Extract sentiment scores and normalize them to range 0-1
    sentiment_scores = {}
    for i, result in enumerate(sentences_with_sentiment):
        # Map sentiment labels to numerical values (assuming POSITIVE=1, NEUTRAL=0, NEGATIVE=-1)
        label = result.get('sentiment_label', 'NEUTRAL')
        label_value = 1 if label == 'POSITIVE' else (-1 if label == 'NEGATIVE' else 0)
        
        # Normalize to 0-1 range (from -1 to 1)
        normalized_sentiment = (label_value + 1) / 2
        
        # Store the sentiment score
        original_index = filtered_indices[i]
        sentiment_scores[original_index] = normalized_sentiment * result.get('sentiment_score', 1.0)
    
    # 8. Similarity matrix
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    threshold = similarity_threshold_english if language == "english" else similarity_threshold_arabic
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix[similarity_matrix < threshold] = 0
    
    # 9. Group sentences by topic
    topic_groups = {}
    for i, topic in enumerate(topics):
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # 10. PageRank and MMR with sentiment analysis integration
    import networkx as nx

    selected_indices = []
    topic_scores = {}

    for topic, indices in topic_groups.items():
        if not indices:
            continue
        
        if len(indices) == 1:
            original_idx = filtered_indices[indices[0]]
            # Apply sentiment score directly for single sentences
            final_score = sentiment_scores.get(original_idx, 1.0)
            selected_indices.append(original_idx)
            topic_scores[original_idx] = final_score
            continue
        
        topic_sim_matrix = np.zeros((len(indices), len(indices)))
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                topic_sim_matrix[i, j] = similarity_matrix[idx1, idx2]
        
        nx_graph = nx.from_numpy_array(topic_sim_matrix)
        try:
            scores = nx.pagerank(nx_graph, alpha=pagerank_alpha)

            for i, idx in enumerate(indices):
                original_idx = filtered_indices[idx]
                duration = durations[original_idx]
                
                # Multiply PageRank score by log duration
                pagerank_score = scores[i] * np.log1p(duration)
                
                # Apply sentiment factor
                sentiment_factor = sentiment_scores.get(original_idx, 1.0)
                
                # Calculate final score as product of PageRank and sentiment
                final_score = pagerank_score * sentiment_factor
                topic_scores[original_idx] = final_score
            
            lambda_val = mmr_lambda_english if language == "english" else mmr_lambda_arabic
            n_to_select = max(1, int(len(indices) * summary_ratio))
            
            selected = []
            unselected = list(range(len(indices)))
            
            # Select first by highest score (now incorporating sentiment)
            first_idx = max(unselected, key=lambda i: topic_scores[filtered_indices[indices[i]]])
            selected.append(first_idx)
            unselected.remove(first_idx)

            while len(selected) < n_to_select and unselected:
                mmr_scores = []
                for i in unselected:
                    relevance = topic_scores[filtered_indices[indices[i]]]
                    diversity = max([topic_sim_matrix[i, j] for j in selected]) if selected else 0
                    mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([filtered_indices[indices[i]] for i in selected])

        except:
            n_to_select = max(1, int(len(indices) * summary_ratio))
            
            # Sort indices by sentiment scores
            sorted_indices = sorted(indices, 
                                   key=lambda idx: sentiment_scores.get(filtered_indices[idx], 0.0), 
                                   reverse=True)
            selected = sorted_indices[:n_to_select]
            selected_indices.extend([filtered_indices[i] for i in selected])
            
            for i in selected:
                original_idx = filtered_indices[i]
                topic_scores[original_idx] = sentiment_scores.get(original_idx, 1.0)
    
    # 11. Sort to maintain order
    selected_indices.sort()

    # 12. Prepare final summary with sentiment information
    summary = []
    
    # Create a mapping of filtered_indices to their positions in the array
    # This helps us look up the correct topic and sentiment info
    idx_to_position = {idx: pos for pos, idx in enumerate(filtered_indices)}
    
    for i, idx in enumerate(selected_indices):
        original_item = sentence_list[idx]
        position = idx_to_position.get(idx)
        
        # Get sentiment information for the sentence
        sentiment_label = "NEUTRAL"
        sentiment_score = 0
        if position is not None and position < len(sentences_with_sentiment):
            sentiment_label = sentences_with_sentiment[position].get('sentiment_label', 'NEUTRAL')
            sentiment_score = sentences_with_sentiment[position].get('sentiment_score', 0)
        
        summary_item = {
            'id': i + 1,
            'sentence': original_item['sentence'],
            'start': original_item['start'],
            'end': original_item['end'],
            'topic': topics[position] if position is not None else 0,  # Use 0 as fallback
            'score': topic_scores.get(idx, 0),
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score
        }
        summary.append(summary_item)
    
    return summary
def sentiment_analyser(chunks):
    """
    Analyze sentiment of each text item in the provided list using XLM-RoBERTa.
    
    Args:
        chunks: List of dictionaries with 'text' key
        
    Returns:
        List of dictionaries with sentiment information added
    """
    from transformers import pipeline
    
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt)
    
    for chunk in chunks:
        sentence = chunk.get('text', '')
        if sentence:
            # Run sentiment analysis on the sentence
            result = pipe(sentence, top_k=1, truncation=True)[0]  # result is a list with one dictionary
            # Add sentiment score to the original dictionary
            chunk['sentiment_score'] = result['score']
            chunk['sentiment_label'] = result['label']
    
    return chunks
from concurrent.futures import ThreadPoolExecutor

def merge_extractive_summaries(
    model_name,
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold_english=0.3,
    similarity_threshold_arabic=0.25,
    pagerank_alpha=0.85,
    mmr_lambda_english=0.6,
    mmr_lambda_arabic=0.7,
    filter_negative_topics=True,
    final_summary_ratio=None  # Optional parameter to control final summary size
):

    # Import the necessary summarization functions

    
    def run_summary_1():
        return generate_extractive_summary_1(
            model_name=model_name,
            sentence_list=sentence_list,
            language=language,
            summary_ratio=summary_ratio,
            min_duration=min_duration,
            min_topic_size=min_topic_size,
            similarity_threshold_english=similarity_threshold_english,
            similarity_threshold_arabic=similarity_threshold_arabic,
            pagerank_alpha=pagerank_alpha,
            mmr_lambda_english=mmr_lambda_english,
            mmr_lambda_arabic=mmr_lambda_arabic,
            filter_negative_topics=filter_negative_topics
        )

    def run_summary_2():
        return generate_extractive_summary_2(
            model_name=model_name,
            sentence_list=sentence_list,
            language=language,
            summary_ratio=summary_ratio,
            min_duration=min_duration,
            min_topic_size=min_topic_size,
            similarity_threshold_english=similarity_threshold_english,
            similarity_threshold_arabic=similarity_threshold_arabic,
            pagerank_alpha=pagerank_alpha,
            mmr_lambda_english=mmr_lambda_english,
            mmr_lambda_arabic=mmr_lambda_arabic,
            filter_negative_topics=filter_negative_topics
        )

    # Run both summary generators in parallel
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(run_summary_1)
        future2 = executor.submit(run_summary_2)
        summary1 = future1.result()
        summary2 = future2.result()
    
    # Early return if either summary is empty
    if not summary1 and not summary2:
        return []
    elif not summary1:
        return summary2
    elif not summary2:
        return summary1
    
    # Create a dictionary to track unique entries by their start/end times
    merged_items = {}
    
    # Process summary1 entries
    for item in summary1:
        time_key = (item['start'], item['end'])
        merged_items[time_key] = {
            'sentence': item['sentence'],
            'start': item['start'],
            'end': item['end'],
            'topic': item['topic'],
            'score': item['score'],
            'source': 'standard'
        }
    
    # Process summary2 entries, adding sentiment info and keeping uniqueness
    for item in summary2:
        time_key = (item['start'], item['end'])
        
        if time_key in merged_items:
            # Update existing entry with sentiment information
            merged_items[time_key].update({
                'sentiment_label': item.get('sentiment_label', 'NEUTRAL'),
                'sentiment_score': item.get('sentiment_score', 0),
                'source': 'both'  # Mark as coming from both summaries
            })
        else:
            # Add new entry from summary2
            merged_items[time_key] = {
                'sentence': item['sentence'],
                'start': item['start'],
                'end': item['end'],
                'topic': item['topic'],
                'score': item['score'],
                'sentiment_label': item.get('sentiment_label', 'NEUTRAL'),
                'sentiment_score': item.get('sentiment_score', 0),
                'source': 'sentiment'
            }
    
    # Convert dictionary back to list and sort by start time
    merged_summary = list(merged_items.values())
    merged_summary.sort(key=lambda x: x['start'])
    
    # Apply final summary ratio if provided
    if final_summary_ratio is not None:
        # Calculate target size
        target_size = max(1, int(len(sentence_list) * final_summary_ratio))
        
        if len(merged_summary) > target_size:
            # Sort by score (higher is better)
            merged_summary.sort(key=lambda x: x['score'], reverse=True)
            merged_summary = merged_summary[:target_size]
            # Resort by start time to maintain chronological order
            merged_summary.sort(key=lambda x: x['start'])
    
    # Reassign IDs for the final summary
    for i, item in enumerate(merged_summary):
        item['id'] = i + 1
    
    return merged_summary