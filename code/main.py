from utils import segment_by_pauses,filter_by_word_count,process_transcript_words, save_extractive_summary,read_word_data
from text_processing import merge_extractive_summaries,chunk_subtitle_segments
from video_processing import process_video
from sklearn.feature_extraction.text import TfidfVectorizer


import time

def generate_highlights(video, language='english'):
    total_start = time.time()  # Track total time

    video_name = video
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path = f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path = f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path = f'video_samples/{video_name}/extracted_summary.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2

    start = time.time()
    words_data = read_word_data(formatted_transcript_path)
    print("read_word_data:", time.time() - start, "seconds")

    if language == 'english':
        start = time.time()
        sentences_timestamps = process_transcript_words(words_data, output_file=formatted_sentences_counts_path)
        print("process_transcript_words:", time.time() - start, "seconds")
    else:
        start = time.time()
        sentences_timestamps = segment_by_pauses(words_data)
        print("segment_by_pauses:", time.time() - start, "seconds")

    start = time.time()
    sentences_timestamps_chunked = chunk_subtitle_segments(model_name, sentences_timestamps, threshold_similarity)
    print("chunk_subtitle_segments:", time.time() - start, "seconds")

    start = time.time()
    sentences_timestamps_chunked = filter_by_word_count(sentences_timestamps_chunked, 6)
    print("filter_by_word_count:", time.time() - start, "seconds")

    start = time.time()
    summary = merge_extractive_summaries(model_name, sentences_timestamps_chunked, summary_ratio=0.25,
                                         min_topic_size=2, mmr_lambda_english=0.7, language=language)
    print("merge_extractive_summaries:", time.time() - start, "seconds")

    start = time.time()
    save_extractive_summary(extracted_summary_path, summary=summary, merged=True)
    print("save_extractive_summary:", time.time() - start, "seconds")

    start = time.time()
    process_video(summary, video_path, output_path)
    print("process_video:", time.time() - start, "seconds")

    total_end = time.time()
    print("Total execution time:", total_end - total_start, "seconds")



if __name__ == "__main__":
    
    generate_highlights('uiC3mhmh8AQ_30minutes',language='english')    
    #uiC3mhmh8AQ_30minutes
    #Ti5vfu9arXQ_25minutes
    #6vX3Us1TOw8_14minutes
    #arj7oStGLkU_14minutes
    #P6FORpg0KVo_12minutes
    #xjGJ5wYs8AQ_10minutes
    #sR6P5Qdvlnk_6minutes