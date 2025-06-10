from core.config import settings
from Optimized_Higlights.utils import process_transcript_words, segment_by_pauses
from Optimized_Higlights.video_processing import process_video


def read_word_data(file_path):
    """
    Reads data from the formatted transcript file and returns a list of dictionaries
    for each word entry.

    Args:
        file_path (str): Path to the formatted transcript file

    Returns:
        list: A list of dictionaries, each containing word metadata (id, start, end, word, probability)
    """
    word_data = []
    current_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if not line:
                continue  # Skip empty lines

            # Parse key-value pairs - find the first colon as the separator
            colon_index = line.find(':')
            if colon_index == -1:
                continue  # Skip lines without a colon

            key = line[:colon_index].strip()
            value = line[colon_index + 1:].strip()

            # Convert value types where necessary
            if key == "id":
                current_data[key] = int(value)
            elif key in ["start", "end", "probability"]:
                current_data[key] = float(value)
            elif key == "word":
                current_data[key] = value

            # When we have all 5 required fields, add the entry to our list
            if len(current_data) == 5:
                word_data.append(current_data.copy())  # Create a copy to avoid reference issues
                current_data = {}  # Reset for the next entry

    return word_data


def save_extractive_summary(path_to_save_at, summary, merged=False):
    """
    Save the extractive summary to a file.
    
    Args:
        path_to_save_at: Path where the summary will be saved
        summary: List of summary segment dictionaries
        merged: Whether the summary is a merged summary with sentiment info
    """
    with open(path_to_save_at, "w", encoding="utf-8") as file:
        for i, item in enumerate(summary, 1):
            file.write(f"id: {i}\n")
            file.write(f"sentence: {item['sentence']}\n")
            file.write(f"start: {item['start']}\n")
            file.write(f"end: {item['end']}\n")
            file.write(f"topic: {item['topic']}\n")
            file.write(f"score: {item['score']}\n")
            
            if merged and 'sentiment_label' in item:
                file.write(f"sentiment_label: {item['sentiment_label']}\n")
                file.write(f"sentiment_score: {item.get('sentiment_score', 0)}\n")
                file.write(f"source: {item.get('source', 'standard')}\n")
                
            file.write('-----------------------------------------------------------\n')
    
    print(f"Extractive summary saved at: {path_to_save_at}")


def convert_transcript_to_word_level(transcript_segments, avg_word_duration=0.3, probability=0.95):
    """
    Convert standard transcript segments [(start, end, text), ...] to word-level format
    for more advanced sentence formation.
    
    Args:
        transcript_segments: List of transcript segments as [(start, end, text), ...]
        avg_word_duration: Average duration of words in seconds for spacing estimation
        probability: Default confidence probability to assign to each word
        
    Returns:
        List of word dictionaries compatible with read_word_data() output format
    """
    word_data = []
    word_id = 1
    
    for start, end, text in transcript_segments:
        # Skip empty segments
        if not text or not text.strip():
            continue
            
        # Split text into words
        words = text.strip().split()
        if not words:
            continue
            
        # Calculate approximate duration for each word
        segment_duration = end - start
        # If we have more than one word, distribute time evenly
        if len(words) > 1:
            word_duration = segment_duration / len(words)
        else:
            word_duration = avg_word_duration
            
        # Create word entries
        current_start = start
        for word in words:
            # Handle very short durations
            current_duration = max(0.1, min(avg_word_duration * 2, 
                                           len(word) * avg_word_duration / 5))
            
            # Don't exceed segment end time
            current_end = min(end, current_start + current_duration)
            
            # Create word entry
            word_data.append({
                "id": word_id,
                "start": current_start,
                "end": current_end,
                "word": word,
                "probability": probability
            })
            
            # Increment word ID and start time for next word
            word_id += 1
            current_start = current_end
    
    return word_data


def save_word_level_transcript(word_data, output_path):
    """
    Save word-level transcript data to a formatted file.
    
    Args:
        word_data: List of word dictionaries
        output_path: Path to save the formatted transcript
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for word in word_data:
            file.write(f"id: {word['id']}\n")
            file.write(f"start: {word['start']}\n")
            file.write(f"end: {word['end']}\n")
            file.write(f"word: {word['word']}\n")
            file.write(f"probability: {word['probability']}\n")
            file.write("\n")
    
    print(f'Word-level transcript saved at: {output_path}')


def preprocess_transcript_segments(transcript_segments):
    """
    Convert transcript segments to dictionary format for processing.
    
    Args:
        transcript_segments: List of transcript segments as [(start, end, text), ...]
        
    Returns:
        List of dictionaries with id, sentence, start, end
    """
    return [
        {
            "id": i,
            "sentence": text,
            "start": start,
            "end": end
        }
        for i, (start, end, text) in enumerate(transcript_segments)
    ]

async def generate_highlights_algorithmically(
    video_path,
    transcript_segments,
    video_info,
    target_duration=None,
    is_reel=False,
    use_advanced_sentence_formation=False
):
    """
    Async-compatible highlight generation function using algorithmic methods.
    """
    total_start = time.time()
    print("Starting highlight generation...")

    # Step 1: Setup directory and safe paths
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(settings.OUTPUT_DIR, video_id)
    os.makedirs(video_dir, exist_ok=True)

    formatted_transcript_path = os.path.join(video_dir, "formatted_transcript.txt")
    formatted_sentences_counts_path = os.path.join(video_dir, "sentences_counts.txt")
    extracted_summary_path = os.path.join(video_dir, "extracted_summary.txt")
    output_path = os.path.join(video_dir, "highlights.mp4")
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    language = 'english'

    # Step 2: Prepare or load word-level transcript
    if not os.path.exists(formatted_transcript_path):
        word_data = convert_transcript_to_word_level(transcript_segments)
        save_word_level_transcript(word_data, formatted_transcript_path)

    words_data = read_word_data(formatted_transcript_path)

    # Step 3: Sentence segmentation
    if language == 'english':
        sentences_timestamps = process_transcript_words(
            words_data,
            output_file=formatted_sentences_counts_path
        )
    else:
        sentences_timestamps = segment_by_pauses(words_data)

    # Step 4: Chunk, filter, summarize
    chunked = chunk_subtitle_segments(
        sentences_timestamps, similarity_threshold=threshold_similarity, model_name=model_name
    )
    chunked = filter_by_word_count(chunked, min_word_count=6)

    summary = merge_extractive_summaries(
        chunked,
        language=language,
        summary_ratio=0.25,
        min_topic_size=2,
        mmr_lambda=0.7,
        model_name=model_name
    )

    # Step 5: Save extractive summary
    save_extractive_summary(extracted_summary_path, summary=summary, merged=True)

    # Step 6: Generate video
    process_video(summary, video_path, output_path)

    # Step 7: Format output for frontend
    highlight_segments = [
        {
            "start": item["start"],
            "end": item["end"],
            "description": item.get("sentence", f"Segment {i + 1}")
        }
        for i, item in enumerate(summary)
    ]

    total_end = time.time()
    print(f"Highlight generation complete in {total_end - total_start:.2f}s")
    
    return output_path, highlight_segments

        
import os
import time
import re
import math
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from moviepy import VideoFileClip, concatenate_videoclips
from logger_config import logger

# Try to import BERTopic, with graceful fallback if not available
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    print("BERTopic not available. Will use fallback methods for topic modeling.")
    BERTOPIC_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available. Will skip sentiment analysis.")
    TRANSFORMERS_AVAILABLE = False


def get_embeddings(model_name, all_sentences):
    """Generate embeddings for a list of sentences."""
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(all_sentences, normalize_embeddings=True)
    return embeddings


def chunk_subtitle_segments(sentences_data, similarity_threshold=0.2, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Create chunks of similar sentences based on semantic similarity.
    
    Args:
        sentences_data: List of dictionaries with 'sentence', 'start', 'end' keys
        similarity_threshold: Threshold value for similarity to consider sentences as part of the same chunk
        model_name: Name of the sentence transformer model to use
        
    Returns:
        List of chunks with text, start time, end time
    """
    print("Starting context-based chunking with metadata preservation...")
    
    # Extract sentences
    all_sentences = [item['sentence'] for item in sentences_data]
    num_sentences = len(all_sentences)
    
    if num_sentences == 0:
        return []
    
    print(f"Total sentences: {num_sentences}")
    
    # Generate embeddings
    print("Loading model and encoding...")
    embeddings = get_embeddings(model_name, all_sentences)
    
    # Create chunks based on similarity
    chunks = []
    current_chunk_indices = [0]
    current_chunk_sentences = [all_sentences[0]]
    
    for i in range(1, num_sentences):
        last_index = current_chunk_indices[-1]
        # Calculate similarity between current sentence and last added sentence
        similarity = cosine_similarity([embeddings[last_index]], [embeddings[i]])[0][0]
        
        if similarity >= similarity_threshold:
            # Add to current chunk
            current_chunk_indices.append(i)
            current_chunk_sentences.append(all_sentences[i])
        else:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            
            # Get start time from first sentence and end time from last sentence in the chunk
            chunk_start = sentences_data[current_chunk_indices[0]]["start"]
            chunk_end = sentences_data[current_chunk_indices[-1]]["end"]
            
            chunks.append({
                "sentence": chunk_text,
                "start": chunk_start,
                "end": chunk_end
            })
            
            # Start a new chunk
            current_chunk_indices = [i]
            current_chunk_sentences = [all_sentences[i]]
    
    # Handle the last chunk
    if current_chunk_indices:
        chunk_text = " ".join(current_chunk_sentences)
        chunk_start = sentences_data[current_chunk_indices[0]]["start"]
        chunk_end = sentences_data[current_chunk_indices[-1]]["end"]
        
        chunks.append({
            "sentence": chunk_text,
            "start": chunk_start,
            "end": chunk_end
        })
    
    print(f"Chunking complete. Created {len(chunks)} chunks.")
    return chunks


def filter_by_word_count(dictionaries, min_word_count=6, sentence_key='sentence'):
    """
    Filter dictionaries to only include those with a minimum word count in the specified text field.
    
    Args:
        dictionaries: List of dictionaries containing text to filter
        min_word_count: Minimum word count to include
        sentence_key: Key in the dictionary that contains the text to count words from
        
    Returns:
        List of dictionaries that meet the minimum word count requirement
    """
    filtered_list = []
    
    for item in dictionaries:
        # Skip if the specified key doesn't exist in the dictionary
        if sentence_key not in item:
            continue
            
        # Get the text and count words
        text = item[sentence_key]
        word_count = len(text.split())
        
        # Add to filtered list if word count meets or exceeds threshold
        if word_count >= min_word_count:
            filtered_list.append(item)
    
    print(f"Filtered to {len(filtered_list)} segments with {min_word_count}+ words")
    return filtered_list


def sentiment_analyser(chunks):
    """
    Analyze sentiment of each text item in the provided list using XLM-RoBERTa.
    
    Args:
        chunks: List of dictionaries with 'text' key
        
    Returns:
        List of dictionaries with sentiment information added
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, skipping sentiment analysis")
        return chunks
        
    try:
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        pipe = pipeline("text-classification", model=model_ckpt)
        
        for chunk in chunks:
            # Get the text to analyze (use 'sentence' or 'text' key)
            sentence = chunk.get('text', chunk.get('sentence', ''))
            if sentence:
                # Run sentiment analysis on the sentence
                result = pipe(sentence, top_k=1, truncation=True)[0]
                # Add sentiment score to the original dictionary
                chunk['sentiment_score'] = result['score']
                chunk['sentiment_label'] = result['label']
        
        return chunks
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return chunks


def generate_extractive_summary(
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold=0.3,
    pagerank_alpha=0.85,
    mmr_lambda=0.6,
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    filter_negative_topics=True
):
    """
    Generate extractive summary from a list of sentence dictionaries.

    Args:
        sentence_list: List of dictionaries with keys: 'sentence', 'start', 'end'
        language: Language of text, 'english' or other
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold: Cosine similarity threshold
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda: Lambda for MMR (balance between relevance/diversity)
        model_name: Name of the sentence transformer model to use
        filter_negative_topics: Whether to remove sentences with topic value of -1

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
    
    # 3. Generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(filtered_sentences)
    
    # 4. Topic modeling
    topics = [0] * len(filtered_sentences)  # Default all to topic 0
    
    if BERTOPIC_AVAILABLE and len(filtered_sentences) >= 4:
        try:
            topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size, embedding_model=model_name)
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except Exception as e:
            print(f"Topic modeling error: {e}")
    
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
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix[similarity_matrix < similarity_threshold] = 0
    
    # 7. Group sentences by topic
    topic_groups = {}
    for i, topic in enumerate(topics):
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # 8. PageRank and MMR
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
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([filtered_indices[indices[i]] for i in selected])

        except Exception as e:
            print(f"PageRank/MMR error: {e}")
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
            'topic': topic_map.get(idx, 0),  # Use 0 as fallback
            'score': topic_scores.get(idx, 0)
        }
        summary.append(summary_item)
    
    return summary


def generate_extractive_summary_sentiments(
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold=0.3,
    pagerank_alpha=0.85,
    mmr_lambda=0.6,
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    filter_negative_topics=True
):
    """
    Generate extractive summary with sentiment analysis integration.

    Args:
        sentence_list: List of dictionaries with keys: 'sentence', 'start', 'end'
        language: Language of text, 'english' or other
        summary_ratio: Percentage of sentences to include in summary (per topic)
        min_duration: Minimum duration (in seconds) to keep a subtitle segment
        min_topic_size: Minimum number of sentences in a topic for BERTopic
        similarity_threshold: Cosine similarity threshold
        pagerank_alpha: Damping factor for PageRank algorithm
        mmr_lambda: Lambda for MMR (balance between relevance/diversity)
        model_name: Name of the sentence transformer model to use
        filter_negative_topics: Whether to remove sentences with topic value of -1

    Returns:
        List of dictionaries representing the summary sentences with sentiment information
    """
    # Skip sentiment analysis if transformers is not available
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, falling back to regular extractive summary")
        return generate_extractive_summary(
            sentence_list, language, summary_ratio, min_duration,
            min_topic_size, similarity_threshold, pagerank_alpha,
            mmr_lambda, model_name, filter_negative_topics
        )
    
    # 1. Extract sentences and durations
    sentences = [item['sentence'] for item in sentence_list]
    durations = [item['end'] - item['start'] for item in sentence_list]
    
    # 2. Filter by duration
    filtered_indices = [i for i, duration in enumerate(durations) if duration >= min_duration]
    if not filtered_indices:
        filtered_indices = list(range(len(sentences)))
    
    filtered_sentences = [sentences[i] for i in filtered_indices]
    
    # 3. Generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(filtered_sentences)
    
    # 4. Topic modeling
    topics = [0] * len(filtered_sentences)  # Default all to topic 0
    
    if BERTOPIC_AVAILABLE and len(filtered_sentences) >= 4:
        try:
            topic_model = BERTopic(language="multilingual", min_topic_size=min_topic_size, embedding_model=model_name)
            topics, _ = topic_model.fit_transform(filtered_sentences, embeddings)
        except Exception as e:
            print(f"Topic modeling error: {e}")
    
    # 5. Filter out sentences with topic -1 if requested
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
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix[similarity_matrix < similarity_threshold] = 0
    
    # 9. Group sentences by topic
    topic_groups = {}
    for i, topic in enumerate(topics):
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)
    
    # 10. PageRank and MMR with sentiment analysis integration
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
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity
                    mmr_scores.append((i, mmr_score))
                
                next_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(next_idx)
                unselected.remove(next_idx)
            
            selected_indices.extend([filtered_indices[indices[i]] for i in selected])

        except Exception as e:
            print(f"PageRank/MMR error: {e}")
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


def merge_extractive_summaries(
    sentence_list,
    language="english",
    summary_ratio=0.3,
    min_duration=5.0,
    min_topic_size=2,
    similarity_threshold=0.3,
    pagerank_alpha=0.85,
    mmr_lambda=0.6,
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    filter_negative_topics=True,
    final_summary_ratio=None  # Optional parameter to control final summary size
):
    """
    Merge extractive summaries from two different algorithms for better results.
    
    Args:
        sentence_list: List of dictionaries with sentence information
        language: Language of the content
        summary_ratio: Percentage of sentences to include in summary
        min_duration: Minimum duration for segments
        min_topic_size: Minimum topic size for BERTopic
        similarity_threshold: Threshold for cosine similarity
        pagerank_alpha: Damping factor for PageRank
        mmr_lambda: Lambda value for MMR (balance between relevance and diversity)
        model_name: Name of the sentence transformer model
        filter_negative_topics: Whether to remove sentences with topic -1
        final_summary_ratio: Optional final ratio to control overall summary size
        
    Returns:
        List of summary segments with all metadata merged
    """
    # Define functions to run each summary generator
    def run_summary_1():
        return generate_extractive_summary(
            sentence_list=sentence_list,
            language=language,
            summary_ratio=summary_ratio,
            min_duration=min_duration,
            min_topic_size=min_topic_size,
            similarity_threshold=similarity_threshold,
            pagerank_alpha=pagerank_alpha,
            mmr_lambda=mmr_lambda,
            model_name=model_name,
            filter_negative_topics=filter_negative_topics
        )

    def run_summary_2():
        return generate_extractive_summary_sentiments(
            sentence_list=sentence_list,
            language=language,
            summary_ratio=summary_ratio,
            min_duration=min_duration,
            min_topic_size=min_topic_size,
            similarity_threshold=similarity_threshold,
            pagerank_alpha=pagerank_alpha,
            mmr_lambda=mmr_lambda,
            model_name=model_name,
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


def snap_highlights_to_transcript_boundaries(highlights, transcript, max_shift=2.0):
    """Refine highlight start/end times to align with transcript boundaries."""
    if not transcript:
        return highlights

    # Gather all boundaries from transcript
    boundaries = set()
    for seg in transcript:
        boundaries.add(seg["start"])
        boundaries.add(seg["end"])
    boundary_list = sorted(boundaries)

    def find_nearest_boundary(time):
        """Find nearest boundary within max_shift."""
        best = None
        best_diff = float('inf')
        for b in boundary_list:
            diff = abs(b - time)
            if diff < best_diff:
                best_diff = diff
                best = b
            else:
                if b > time and diff > best_diff:
                    break
        return (best, best_diff)

    snapped = []
    for hl in highlights:
        start_t = hl["start"]
        end_t = hl["end"]
        # Snap start
        best_boundary, diff = find_nearest_boundary(start_t)
        if diff <= max_shift:
            start_t = best_boundary
        # Snap end
        best_boundary, diff = find_nearest_boundary(end_t)
        if diff <= max_shift:
            end_t = best_boundary
        # Ensure valid
        if end_t > start_t:
            snapped.append({
                "start": start_t,
                "end": end_t,
                "description": hl.get("description", hl.get("sentence", "Highlight segment"))
            })
    # Sort again
    snapped.sort(key=lambda x: x["start"])
    return snapped


def extract_highlights(video_path, highlights):
    """Extract each highlight using MoviePy."""
    temp_dir = os.path.join("output", "temp_clips")
    os.makedirs(temp_dir, exist_ok=True)

    # Extract clips using ThreadPoolExecutor for parallel processing
    def extract_clip(idx, segment):
        start = segment["start"]
        end = segment["end"]
        out_file = os.path.join(temp_dir, f"highlight_{idx}.mp4")
        
        try:
            print(f"Extracting clip {idx+1}/{len(highlights)}: {start:.1f}s–{end:.1f}s")
            video = VideoFileClip(video_path)
            start_buffer = max(0, start - 0.2)  # Small buffer for smoother transitions
            clip = video.subclipped(start_buffer, end)
            clip.write_videofile(
                out_file,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast",
            )
            clip.close()
            video.close()

            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                return out_file, segment
            else:
                print(f"Failed to create clip at {out_file} (empty file).")
                return None, None
        except Exception as e:
            print(f"Error extracting clip {idx}: {e}")
            return None, None

    # Process clips in parallel
    successful_clips = []
    highlight_info = []

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        futures = {executor.submit(extract_clip, i, hl): i for i, hl in enumerate(highlights)}
        for future in futures:
            clip_path, info = future.result()
            if clip_path:
                successful_clips.append(clip_path)
                highlight_info.append(info)

    # Check if we have any successful clips
    if not successful_clips:
        # Create a fallback clip if all extractions failed
        fallback_clip = os.path.join(temp_dir, "highlight_default.mp4")
        video = VideoFileClip(video_path)
        fallback_duration = min(30, video.duration)
        clip = video.subclipped(0, fallback_duration)
        clip.write_videofile(
            fallback_clip,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="veryfast",
        )
        clip.close()
        video.close()

        return [fallback_clip], [{
            "start": 0,
            "end": fallback_duration,
            "description": "Default highlight"
        }]

    # Return successful clips
    return successful_clips, highlight_info