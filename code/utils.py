import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


    


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




def segment_by_pauses(word_timestamps, min_pause_duration=0.7):
    sentences = []
    current_sentence = []
    current_start = word_timestamps[0]["start"]
    
    for i in range(1, len(word_timestamps)):
        current_sentence.append(word_timestamps[i-1]["word"])
        
        # Calculate pause between current word and next word
        pause_duration = word_timestamps[i]["start"] - word_timestamps[i-1]["end"]
        
        # If there's a significant pause, mark as sentence boundary
        if pause_duration >= min_pause_duration:
            sentences.append({
                "sentence": " ".join(current_sentence),
                "start": current_start,
                "end": word_timestamps[i-1]["end"]
            })
            current_sentence = []
            current_start = word_timestamps[i]["start"]
    
    # Add the last sentence
    if current_sentence:
        sentences.append({
            "sentence": " ".join(current_sentence),
            "start": current_start,
            "end": word_timestamps[-1]["end"]
        })
    
    return sentences