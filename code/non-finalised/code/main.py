from utils import filter_by_word_count,process_transcript_words, save_extractive_summary,process_chunks_by_topic,group_by_topic_successively,read_and_split_sentences,read_word_data,build_sentence_metadata,sort_chunks,select_chunks,filter_by_duration,save_formatted_sentences_counts
from text_processing import merge_extractive_summaries,generate_extractive_summary_2,topic_based_chunking_and_summarization_with_sentiments,topic_based_chunking_and_summarization,generate_extractive_summary_1,chunk_subtitle_segments,get_importance_textRank,sentences_with_topic_modelling
from video_processing import process_video
from sklearn.feature_extraction.text import TfidfVectorizer

def sequential_chunking_approach():
    video_name = '6vX3Us1TOw8_14minutes'
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    minimum_duration=0
    selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    
    sentences_with_counts=read_and_split_sentences(full_transcript_path)
    save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    
    
    # Step 2: Process text into meaningful chunks

    chunks = chunk_subtitle_segments(model_name, senteces_with_timestamps, threshold_similarity)
    
    
    chunks=filter_by_duration(chunks,minimum_duration)
    sorted_chunks=sort_chunks(chunks)

    
    selected_chunks = select_chunks(sorted_chunks, selection_percentage)
    print('------------------------selected chunks---------------------------------')
    for i in selected_chunks:
        print(i)

    print('----------------------------------------processing video------------------------------------')
    process_video(selected_chunks, video_path, output_path)

def topic_modelling_approach():
   
    video_name = '6vX3Us1TOw8_14minutes'
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    minimum_duration=5
    selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    
    sentences_with_counts=read_and_split_sentences(full_transcript_path)
    save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    sentences_timestamps_topics=sentences_with_topic_modelling(senteces_with_timestamps)
    # for i in senteces_with_timestamps:
    #     print(i)
    sentences_timestamps_topics_grouped=group_by_topic_successively(sentences_timestamps_topics)
    # for i in sentences_timestamps_topics_grouped:
    #     print(i)
    
    sentences_timestamps_topics_grouped_processed=process_chunks_by_topic(sentences_timestamps_topics_grouped,minimum_duration,model_name
                                                                          ,selection_percentage)
    save_extractive_summary(extracted_summary_path,sentences_timestamps_topics_grouped_processed)
    process_video(sentences_timestamps_topics_grouped_processed,video_path,output_path)

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
def claude_suggested_approach(video,language='english'):
    video_name = video
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary_1.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights_1.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    # minimum_duration=5
    # selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    if language=='english':
        sentences_timestamps=process_transcript_words(words_data,output_file=formatted_sentences_counts_path)
    else:
        sentences_timestamps=segment_by_pauses(words_data)
    sentences_timestamps_chunked=chunk_subtitle_segments(model_name,sentences_timestamps,threshold_similarity)
    # sentences_with_counts=read_and_split_sentences(full_transcript_path)
    # save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    sentences_timestamps_chunked=filter_by_word_count(sentences_timestamps_chunked,6)
    
    # senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    summary=generate_extractive_summary_1(model_name,sentences_timestamps_chunked,summary_ratio=0.25,mmr_lambda_english=0.7,language=language)
    save_extractive_summary(extracted_summary_path,summary=summary)
    # process_video(summary,video_path,output_path)
def cluade_suugested_approach_modified(video,language='english'):
    video_name = video
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary_2.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights_2.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    # minimum_duration=5
    # selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    if language=='english':
        sentences_timestamps=process_transcript_words(words_data,output_file=formatted_sentences_counts_path)
    else:
        sentences_timestamps=segment_by_pauses(words_data)
    sentences_timestamps_chunked=chunk_subtitle_segments(model_name,sentences_timestamps,threshold_similarity)
    sentences_timestamps_chunked=filter_by_word_count(sentences_timestamps_chunked,6)

    # sentences_with_counts=read_and_split_sentences(full_transcript_path)
    # save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    # senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    summary=generate_extractive_summary_2(model_name,sentences_timestamps_chunked,summary_ratio=0.25,mmr_lambda_english=0.7,language=language)
    save_extractive_summary(extracted_summary_path,summary=summary)
    # process_video(summary,video_path,output_path)
def merged_summaries(video,language='english'):
    video_name = video
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary_3.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights_3.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    # minimum_duration=5
    # selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    if language=='english':
        sentences_timestamps=process_transcript_words(words_data,output_file=formatted_sentences_counts_path)
    else:
        sentences_timestamps=segment_by_pauses(words_data)
    sentences_timestamps_chunked=chunk_subtitle_segments(model_name,sentences_timestamps,threshold_similarity)
    sentences_timestamps_chunked=filter_by_word_count(sentences_timestamps_chunked,6)
    
    # sentences_with_counts=read_and_split_sentences(full_transcript_path)
    # save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    # senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    summary=merge_extractive_summaries(model_name,sentences_timestamps_chunked,summary_ratio=0.25,min_topic_size=2,mmr_lambda_english=0.7,language=language)
    save_extractive_summary(extracted_summary_path,summary=summary,merged=True)
    process_video(summary,video_path,output_path)
def topic_based_chunking_summarisation():
    video_name = 'P6FORpg0KVo_12minutes'
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary_1.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights_1.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    # minimum_duration=5
    # selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    sentences_timestamps=process_transcript_words(words_data,output_file=formatted_sentences_counts_path)
    # sentences_timestamps_chunked=chunk_subtitle_segments(model_name,sentences_timestamps,threshold_similarity)
    # sentences_with_counts=read_and_split_sentences(full_transcript_path)
    # save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    # senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    summary=topic_based_chunking_and_summarization(model_name,sentences_timestamps,summary_ratio=0.2,mmr_lambda_english=0.7)
    save_extractive_summary(extracted_summary_path,summary=summary)
    process_video(summary,video_path,output_path)
def topic_based_chunking_summarisation_sentiments():
    video_name = 'P6FORpg0KVo_12minutes'
    formatted_transcript_path = f'video_samples/{video_name}/formatted_transcript.txt'
    full_transcript_path=f'video_samples/{video_name}/full_transcript.txt'
    formatted_sentences_counts_path=f'video_samples/{video_name}/sentences_counts.txt'
    extracted_summary_path=f'video_samples/{video_name}/extracted_summary_3.txt'
    video_path = f'video_samples/{video_name}/downloaded_video.mp4'
    output_path = f"video_samples/{video_name}/highlights_3.mp4"
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    threshold_similarity = 0.2
    # minimum_duration=5
    # selection_percentage=0.2

    words_data=read_word_data(formatted_transcript_path)
    sentences_timestamps=process_transcript_words(words_data,output_file=formatted_sentences_counts_path)
    # sentences_timestamps_chunked=chunk_subtitle_segments(model_name,sentences_timestamps,threshold_similarity)
    # sentences_with_counts=read_and_split_sentences(full_transcript_path)
    # save_formatted_sentences_counts(formatted_sentences_counts_path,sentences_with_counts)
    
    
    # senteces_with_timestamps=build_sentence_metadata(sentences_with_counts,words_data)
    summary=topic_based_chunking_and_summarization_with_sentiments(model_name,sentences_timestamps,summary_ratio=0.3,mmr_lambda_english=0.7)
    save_extractive_summary(extracted_summary_path,summary=summary)
    process_video(summary,video_path,output_path)
if __name__ == "__main__":
    
    merged_summaries('arabic_Y1HfRhfHwUc_36minutes',language='arabic')    
    #uiC3mhmh8AQ_30minutes
    #Ti5vfu9arXQ_25minutes
    #6vX3Us1TOw8_14minutes
    #arj7oStGLkU_14minutes
    #P6FORpg0KVo_12minutes
    #xjGJ5wYs8AQ_10minutes
    #sR6P5Qdvlnk_6minutes