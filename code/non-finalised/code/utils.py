import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from text_processing import get_embeddings,get_importance_textRank
def print_chunks_full(chunks):
    """
    Print chunks with complete text content (no truncation)
    """
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*50}")
        print(f"Chunk {i}:")
        print(f"{'='*50}")
        print(f"Start: {chunk['start']:.2f}s, End: {chunk['end']:.2f}s, Score: {chunk['score']:.6f}")
        print(f"\nText:\n{chunk['text']}")

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
            current["tokens"].append(next_seg['tokens'])
            current["avg_logprob"] = (current["avg_logprob"] + next_seg["avg_logprob"]) / 2
            current["compression_ratio"] = (current["compression_ratio"] + next_seg["compression_ratio"]) / 2
            current["no_speech_prob"] = max(current["no_speech_prob"], next_seg["no_speech_prob"])

    # Add the last combined segment
    if current:
        merged.append(current)
    print("completed merging...........")
    return merged

def sort_chunks(chunks,base_key='score'):
    """
    sort the chunks.
    
    Args:
        chunks: List of chunks with timing and score information
        
    Returns:
        List of sorted chunks based on the base_key 
    """
    # Sort chunks by score in descending order

    sorted_chunks = sorted(chunks, key=lambda x: x.get(f'{base_key}', 0), reverse=True)
    return sorted_chunks
    
def select_chunks(chunks, selection_percentage):
        # Select top percentage of chunks
    top_chunks_count = max(1, int(len(chunks) * selection_percentage))
    top_chunks = chunks[:top_chunks_count]
    print(f'Selected top {top_chunks_count} chunks ({selection_percentage*100}%):')
    
    # Sort the top chunks by their start time to maintain chronological order
    chronological_chunks = sorted(top_chunks, key=lambda x: x.get('start', 0))
    
    print(f"Selected {len(chronological_chunks)} chunks out of {len(chunks)}")
    
    return chronological_chunks    

def merge_word_lists(segments):
    """Merges a list of lists of word dictionaries into a single list of word dictionaries."""
    merged_list = []
    for sublist in segments:
        merged_list.extend(sublist['words'])
    return merged_list

def saving_full_transcript(path_to_save_at, text):
    import json
    with open(path_to_save_at, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Full transcript saved to: {path_to_save_at}")



def saving_formatted_transcript(path_to_save_at, list_of_words):
    with open(path_to_save_at, "w", encoding="utf-8") as file:
        for i,word in enumerate(list_of_words,1):
            file.write(f"id: {i}\n")
            file.write(f"start: {word['start']}\n")
            file.write(f"end: {word['end']}\n")
            file.write(f"word: {word['word']}\n")
            file.write(f"probability: {word['probability']}\n")
    print('formatted transcript saved at: ', f'{path_to_save_at}')

def read_and_split_sentences(file_path):
    """
    Analyze a transcript text file, splitting it into sentences and counting words.
    Returns a list of dictionaries with sentence text and word count.

    This version is specifically optimized for transcript-style text where
    traditional NLP sentence boundary detection might struggle.
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Method 1: Regex-based sentence splitting (better for transcripts)
    # Split on sentence endings (.!?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', raw_text)

    # Process each sentence
    sentence_data = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:  # Skip empty sentences
            continue

        # Count words (split on whitespace and filter out punctuation-only tokens)
        words = [w for w in re.split(r'\s+', sent)
                if w and not all(c in ',.!?;:()[]{}"\'' for c in w)]

        sentence_data.append({
            'sentence': sent,
            'count': len(words)
        })

    return sentence_data


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


def build_sentence_metadata(sentences_with_counts, words_data):
    """
    Builds a list of dictionaries containing sentence metadata using word timing information.

    Parameters:
        sentences_with_counts (list): List of dicts like {'sentence': ..., 'count': ...}
        words_data (list): List of dicts like {'id': ..., 'start': ..., 'end': ..., 'word': ..., 'probability': ...}

    Returns:
        list: List of dicts like {'id': ..., 'sentence': ..., 'start': ..., 'end': ...}
    """
    print(len(sentences_with_counts))
    print(len(words_data))
    sentence_metadata = []
    word_index = 0  # pointer to track words_data

    for idx, item in enumerate(sentences_with_counts):
        count = item['count']
        sentence = item['sentence']
        # print('sentence id:' ,idx)
        # print('sentence:  ',sentence)
        # print('sentence length',count)
        # print('from word index: ', word_index)
        # print('to word index: ', word_index+count)
        # print('-----------------------------------------------------')
        if word_index + count > len(words_data):
            break  # Avoid index error if sentence count exceeds word list

        start_time = words_data[word_index]['start']
        end_time = words_data[word_index + count - 1]['end']

        sentence_metadata.append({
            'id': idx + 1,
            'sentence': sentence,
            'start': start_time,
            'end': end_time
        })

        word_index += count  # move pointer forward

    return sentence_metadata

def filter_by_duration(chunks, min_duration_seconds):
    """
    Filter chunks to only include those with a minimum duration.
    
    Args:
        chunks: List of chunk dictionaries with 'start' and 'end' keys
        min_duration_seconds: Minimum duration in seconds
        
    Returns:
        List of chunk dictionaries that meet the minimum duration requirement
    """
    filtered_chunks = []
    
    for chunk in chunks:
        # Calculate duration
        start_time = chunk.get('start', 0)
        end_time = chunk.get('end', 0)
        duration = end_time - start_time
        
        # Include chunk if it meets minimum duration
        if duration >= min_duration_seconds:
            filtered_chunks.append(chunk)
    
    print(f"Filtered {len(chunks) - len(filtered_chunks)} chunks below {min_duration_seconds}s duration.")
    print(f"Retained {len(filtered_chunks)} chunks.")
    
    return filtered_chunks
def save_formatted_sentences_counts(path_to_save_at,sentences_counts):
        with open(path_to_save_at, "w", encoding="utf-8") as file:
            for i,item in enumerate(sentences_counts,1):
                file.write(f"id: {i}\n")
                file.write(f"sentence: {item['sentence']}\n")
                file.write(f"count: {item['count']}\n")
                file.write('-----------------------------------------------------------')
        print(f'formatted sentences and their counts saved at: {path_to_save_at}')
from collections import defaultdict

def group_by_topic_successively(dicts):
    topic_groups = defaultdict(list)
    
    # Group them first
    for d in dicts:
        topic = d.get('topic')
        topic_groups[topic].append(d)
    # Print the length of each topic
    for topic_name, topic_items in topic_groups.items():
        print(f'Length of topic: {topic_name} is {len(topic_items)}')
    # Now flatten the groups into a single list
    grouped_list = []
    for topic in topic_groups:
        grouped_list.extend(topic_groups[topic])
    
    return grouped_list

def process_chunks_by_topic(grouped_chunks, min_duration_seconds, model_name, selection_percentage):
    """
    Process chunks by topic, performing filtering, embedding, importance scoring, sorting, and selection.
    
    Args:
        grouped_chunks: List of dictionaries grouped by topic from group_by_topic_successively
        min_duration_seconds: Minimum duration in seconds for filtering
        model_name: Name of the Sentence Transformer model to use
        selection_percentage: Percentage of top chunks to select per topic
        
    Returns:
        List of selected chunks across all topics
    """
    # Create a dictionary to organize chunks by topic
    topic_dict = defaultdict(list)
    for chunk in grouped_chunks:
        topic = chunk.get('topic')
        topic_dict[topic].append(chunk)
    
    final_selected_chunks = []
    
    # Process each topic separately
    for topic, topic_chunks in topic_dict.items():
        print(f"\nProcessing topic: {topic} with {len(topic_chunks)} chunks")
        
        # Step 1: Filter by duration
        filtered_chunks = filter_by_duration(topic_chunks, min_duration_seconds)
        if not filtered_chunks:
            print(f"No chunks remain for topic {topic} after duration filtering. Skipping.")
            continue
        
        # Step 2: Embed sentences
        sentences = [chunk['sentence'] for chunk in filtered_chunks]
        embeddings = get_embeddings(model_name, sentences)
        
        # Step 3: Calculate importance with TextRank
        importance_scores = get_importance_textRank(embeddings, filtered_chunks)
        
        # Add importance scores to chunks
        for idx, chunk in enumerate(filtered_chunks):
            chunk['score'] = importance_scores[idx]
        
        # Step 4: Sort by importance score
        sorted_chunks = sort_chunks(filtered_chunks, base_key='score')
        
        # Step 5: Select top chunks
        selected_chunks = select_chunks(sorted_chunks, selection_percentage)
        
        # Add to final results
        final_selected_chunks.extend(selected_chunks)
    
    # Sort the final combined list by start time to maintain chronological order
    final_chronological_chunks = sorted(final_selected_chunks, key=lambda x: x.get('start', 0))
    
    print(f"\nFinal result: {len(final_chronological_chunks)} chunks selected across all topics")
    return final_chronological_chunks

def save_extractive_summary(path_to_save_at,summary,merged=False):
        with open(path_to_save_at, "w", encoding="utf-8") as file:
            for i,item in enumerate(summary,1):
                file.write(f"id: {i}\n")
                file.write(f"sentence: {item['sentence']}\n")
                file.write(f"start: {item['start']}")
                file.write(f"end: {item['end']}")
                file.write(f"topic: {item['topic']}")
                file.write(f"score: {item['score']}")
                # if merged:
                #     file.write(f'sentiment label {item['sentiment_label']}')
                #     file.write(f'sentiment score {item['sentiment_score']}')
                #     file.write(f'source  {item['sentiment_label']}')
                
                file.write('-----------------------------------------------------------')
        print(f'extracted summary saved at: {path_to_save_at}')

def process_transcript_words(word_data, probability_threshold=0.5, output_file=None):
    """
    Process word-level timestamp transcript data from Whisper and construct sentences
    
    Args:
        word_data (list): List of word dictionaries with timestamp and probability data
        probability_threshold (float): Minimum probability threshold for including words (0-1)
        output_file (str, optional): Path to save the output text file. If None, won't save to file.
        
    Returns:
        list: List of sentence dictionaries with text, start/end timestamps, and IDs
    """
    # Filter out words below probability threshold
    filtered_words = [word for word in word_data if word.get('probability', 0) >= probability_threshold]
    
    if not filtered_words:
        return []
    
    # Initialize sentence tracking variables
    sentences = []
    current_sentence = {
        'id': filtered_words[0]['id'],
        'sentence': '',
        'words': [],
        'start': filtered_words[0]['start'],
        'end': None,
        'word_ids': [],
        'word_count': 0  # Add word count field
    }
    
    # Sentence-ending punctuation
    sentence_enders = ['.', '!', '?']
    
    # Process each word that passed the probability threshold
    for i, word in enumerate(filtered_words):
        trimmed_word = word['word'].strip()
        if not trimmed_word:
            continue
        
        # Add to current sentence
        current_sentence['words'].append(word)
        current_sentence['word_ids'].append(word['id'])
        current_sentence['word_count'] += 1  # Increment word count
        
        # Add appropriate spacing between words
        if current_sentence['sentence'] and not current_sentence['sentence'].endswith(' '):
            current_sentence['sentence'] += ' '
        
        current_sentence['sentence'] += trimmed_word
        current_sentence['end'] = word['end']
        
        # Check if this word ends a sentence or is the last word
        ends_with_punctuation = any(trimmed_word.endswith(ender) for ender in sentence_enders)
        is_last_word = i == len(filtered_words) - 1
        
        if ends_with_punctuation or is_last_word:
            # Finalize the current sentence
            sentences.append(current_sentence.copy())
            
            # Start a new sentence if there are more words
            if not is_last_word:
                current_sentence = {
                    'id': filtered_words[i + 1]['id'],
                    'sentence': '',
                    'words': [],
                    'start': filtered_words[i + 1]['start'],
                    'end': None,
                    'word_ids': [],
                    'word_count': 0  # Reset word count for new sentence
                }
    
    # Write output to text file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, sentence in enumerate(sentences):
                    f.write(f"Sentence {i+1}:\n")
                    f.write(f"ID: {sentence['id']}\n")
                    f.write(f"sentence: {sentence['sentence']}\n")
                    f.write(f"Word Count: {sentence['word_count']}\n")
                    f.write(f'start:{sentence['start']}\n')
                    f.write(f'end:{sentence['end']}\n')

                    f.write(f"Word IDs: {', '.join(map(str, sentence['word_ids']))}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    
    return sentences

def filter_by_word_count(dictionaries, min_word_count, sentence_key='sentence'):

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
    
    return filtered_list


