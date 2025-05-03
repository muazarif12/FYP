# import json
# import re
# import os
# import asyncio
# from logger_config import logger

# # Import your existing dependencies
# import ollama
# from transcriber import get_youtube_transcript, transcribe_video
# from summarizer import generate_response_async
# from utils import format_timestamp


# async def generate_study_guide(transcript_segments, video_info):
#     """
#     Generate a comprehensive study guide for educational content.
    
#     Args:
#         transcript_segments: List of (start_time, end_time, text) tuples
#         video_info: Dictionary with video metadata
        
#     Returns:
#         A structured study guide with questions, answers, and other educational materials
#     """
#     # Extract full text from transcript
#     full_text = " ".join([seg[2] for seg in transcript_segments])
    
#     # Step 1: Analyze content to determine if it's educational/instructional
#     print("Analyzing video content for educational value...")
    
#     content_analysis_prompt = f"""
#     Analyze this video transcript to determine if it is educational/instructional content suitable for a study guide.
    
#     VIDEO TITLE: {video_info.get('title', 'Unknown')}
#     VIDEO DESCRIPTION: {video_info.get('description', 'No description')}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:3000]}... [truncated]
    
#     Please analyze:
#     1. Is this educational content that would benefit from a study guide? (Yes/No)
#     2. What is the main subject or topic?
#     3. What kind of educational format is it? (lecture, tutorial, talk, presentation, etc.)
#     4. What academic level is it appropriate for? (K-12, undergraduate, graduate, professional)
#     5. What are 3-5 key learning objectives someone might have from this content?
    
#     Return only a JSON object with your analysis:
#     {{
#         "is_educational": true or false,
#         "main_subject": "Subject name",
#         "format": "Format type",
#         "academic_level": "Level",
#         "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"]
#     }}
#     """
    
#     try:
#         content_analysis_response = await generate_response_async(content_analysis_prompt)
        
#         # Extract JSON data
#         json_match = re.search(r'\{[\s\S]*\}', content_analysis_response)
#         if json_match:
#             content_analysis = json.loads(json_match.group(0))
#         else:
#             # Default values if JSON extraction fails
#             content_analysis = {
#                 "is_educational": True,  # Assume educational by default
#                 "main_subject": video_info.get('title', 'Unknown Subject'),
#                 "format": "Presentation",
#                 "academic_level": "General",
#                 "learning_objectives": ["Understanding the main concepts presented in the video"]
#             }
            
#         # If content is not educational, still generate a basic study guide but note limitations
#         if not content_analysis.get("is_educational", True):
#             print("Note: This content may not be primarily educational, but creating a study guide anyway.")
    
#     except Exception as e:
#         print(f"Error analyzing content: {e}")
#         # Default values if analysis fails
#         content_analysis = {
#             "is_educational": True,
#             "main_subject": video_info.get('title', 'Unknown Subject'),
#             "format": "Presentation",
#             "academic_level": "General",
#             "learning_objectives": ["Understanding the main concepts presented in the video"]
#         }
    
#     # Step 2: Generate the study guide sections
#     print("Generating comprehensive study guide sections...")
    
#     # Generate study guide components in parallel for efficiency
#     tasks = [
#         generate_quiz_questions(full_text, content_analysis),
#         generate_quiz_answers(full_text, content_analysis),
#         generate_essay_questions(full_text, content_analysis),
#         generate_key_terms_glossary(full_text, content_analysis),
#         generate_section_summaries(transcript_segments, content_analysis)
#     ]
    
#     quiz_questions, quiz_answers, essay_questions, glossary, section_summaries = await asyncio.gather(*tasks)
    
#     # Step 3: Generate a title for the study guide
#     study_guide_title_prompt = f"""
#     Create an engaging, descriptive title for a study guide based on this video content.
#     The title should be concise (5-10 words) and clearly indicate the main subject.
    
#     VIDEO TITLE: {video_info.get('title', 'Unknown')}
#     MAIN SUBJECT: {content_analysis.get('main_subject', 'Unknown')}
#     LEARNING OBJECTIVES: {', '.join(content_analysis.get('learning_objectives', ['Understanding the content']))}
    
#     Return only the title with no other text.
#     """
    
#     study_guide_title = await generate_response_async(study_guide_title_prompt)
#     study_guide_title = study_guide_title.strip().replace('"', '').replace("'", "")
#     if ":" not in study_guide_title:
#         study_guide_title += f": A Study Guide Based on {video_info.get('title', 'the Video')}"
    
#     # Step 4: Assemble the complete study guide
#     study_guide = {
#         "title": study_guide_title,
#         "video_info": {
#             "title": video_info.get('title', 'Unknown'),
#             "description": video_info.get('description', 'No description')
#         },
#         "content_analysis": content_analysis,
#         "quiz_questions": quiz_questions,
#         "quiz_answers": quiz_answers,
#         "essay_questions": essay_questions,
#         "glossary": glossary,
#         "section_summaries": section_summaries
#     }
    
#     # Step 5: Format the study guide for display
#     formatted_study_guide = format_study_guide_for_display(study_guide)
    
#     # Step 6: Save the study guide to a file
#     file_path = await save_study_guide(formatted_study_guide, video_info.get('title', 'study_guide'))
    
#     return {
#         "study_guide": study_guide,
#         "formatted_study_guide": formatted_study_guide,
#         "file_path": file_path
#     }


# async def generate_quiz_questions(full_text, content_analysis):
#     """Generate a set of quiz questions based on the video content."""
#     print("Generating quiz questions...")
    
#     quiz_prompt = f"""
#     You are an experienced educator. Create 10 well-crafted quiz questions based on this video transcript.
    
#     SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
#     ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:4000]}... [truncated]
    
#     Instructions:
#     1. Create questions that assess understanding of key concepts, not just recall of minor details
#     2. Include a mix of question types (factual recall, concept application, analysis)
#     3. Focus on the most important points from the video
#     4. Ensure questions are clear, unambiguous, and answerable from the transcript
#     5. Phrase questions in a direct, concise manner
#     6. Number each question from 1-10
    
#     Return ONLY the questions without answers or explanations, one per line, numbered 1-10.
#     """
    
#     response = await generate_response_async(quiz_prompt)
    
#     # Clean up the response
#     questions = []
#     current_question = ""
    
#     for line in response.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check if line starts with a number (1-10)
#         if re.match(r'^\d{1,2}[\.\)]', line):
#             # If we have a current question, save it
#             if current_question:
#                 questions.append(current_question)
#             # Start a new question
#             current_question = line
#         else:
#             # Continue the current question
#             current_question += " " + line
    
#     # Add the last question
#     if current_question:
#         questions.append(current_question)
    
#     # Ensure we have questions (if parsing failed, use the whole response)
#     if not questions:
#         questions = [response]
    
#     return questions


# async def generate_quiz_answers(full_text, content_analysis):
#     """Generate answers for the quiz questions."""
#     print("Generating quiz answers...")
    
#     answers_prompt = f"""
#     You are an experienced educator. Create a comprehensive answer key for a 10-question quiz about this video.
    
#     SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
#     ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:4000]}... [truncated]
    
#     Instructions:
#     1. First, generate 10 likely questions that would appear on a quiz about this content
#     2. Then, provide detailed answers (3-5 sentences each) that thoroughly explain the correct response
#     3. Include relevant examples or supporting details from the transcript
#     4. Ensure answers are comprehensive enough to serve as a study guide
#     5. Number each answer from 1-10 to match the questions
    
#     Format each answer as "X. [Detailed answer explanation]" where X is the question number.
#     Return ONLY the answers, one per numbered point.
#     """
    
#     response = await generate_response_async(answers_prompt)
    
#     # Parse answers with the same approach as questions
#     answers = []
#     current_answer = ""
    
#     for line in response.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check if line starts with a number (1-10)
#         if re.match(r'^\d{1,2}[\.\)]', line):
#             # If we have a current answer, save it
#             if current_answer:
#                 answers.append(current_answer)
#             # Start a new answer
#             current_answer = line
#         else:
#             # Continue the current answer
#             current_answer += " " + line
    
#     # Add the last answer
#     if current_answer:
#         answers.append(current_answer)
    
#     # Ensure we have answers (if parsing failed, use the whole response)
#     if not answers:
#         answers = [response]
    
#     return answers


# async def generate_essay_questions(full_text, content_analysis):
#     """Generate thought-provoking essay questions based on the content."""
#     print("Generating essay questions...")
    
#     essay_prompt = f"""
#     You are an experienced educator. Create 5 thought-provoking essay questions based on this video transcript.
    
#     SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
#     ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
#     LEARNING OBJECTIVES: {', '.join(content_analysis.get('learning_objectives', ['Understanding the content']))}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:4000]}... [truncated]
    
#     Instructions:
#     1. Create essay questions that require critical thinking, analysis, evaluation, or synthesis
#     2. Questions should prompt in-depth exploration of the main concepts from the video
#     3. Each question should be suitable for a 1-2 page response
#     4. Focus on controversial, nuanced, or complex aspects of the topic
#     5. Each question should open with a relevant action verb (Analyze, Evaluate, Compare, etc.)
    
#     Return only the 5 essay questions, one per line. Number them 1-5.
#     """
    
#     response = await generate_response_async(essay_prompt)
    
#     # Parse the essay questions
#     essay_questions = []
#     current_question = ""
    
#     for line in response.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check if line starts with a number (1-5)
#         if re.match(r'^\d{1}[\.\)]', line):
#             # If we have a current question, save it
#             if current_question:
#                 essay_questions.append(current_question)
#             # Start a new question
#             current_question = line
#         else:
#             # Continue the current question
#             current_question += " " + line
    
#     # Add the last question
#     if current_question:
#         essay_questions.append(current_question)
    
#     # Ensure we have questions (if parsing failed, use the whole response)
#     if not essay_questions:
#         essay_questions = [response]
    
#     return essay_questions


# async def generate_key_terms_glossary(full_text, content_analysis):
#     """Generate a glossary of key terms from the video content."""
#     print("Generating key terms glossary...")
    
#     glossary_prompt = f"""
#     You are an educational content creator. Create a glossary of 8-12 important key terms used in this video transcript.
    
#     SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
#     ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:4000]}... [truncated]
    
#     Instructions:
#     1. Identify specialized terminology, jargon, or important concepts mentioned in the video
#     2. For each term, provide a clear, concise definition (1-2 sentences) based on how it was used in the video
#     3. Focus on terms that are central to understanding the main concepts
#     4. Format as "Term: Definition"
#     5. Include only terms that actually appear in the transcript
    
#     Return ONLY the glossary entries, one term per line, in alphabetical order.
#     """
    
#     response = await generate_response_async(glossary_prompt)
    
#     # Parse the glossary entries
#     glossary_entries = []
#     current_entry = ""
    
#     for line in response.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         # Check if line contains a term definition (Term: Definition)
#         if re.match(r'^[A-Za-z\s\(\)]+:', line):
#             # If we have a current entry, save it
#             if current_entry:
#                 glossary_entries.append(current_entry)
#             # Start a new entry
#             current_entry = line
#         else:
#             # Continue the current entry
#             current_entry += " " + line
    
#     # Add the last entry
#     if current_entry:
#         glossary_entries.append(current_entry)
    
#     # Ensure we have entries (if parsing failed, try alternative parsing)
#     if not glossary_entries:
#         # Try alternative parsing - look for bold/emphasized terms
#         glossary_entries = re.findall(r'\*\*([^*]+)\*\*:([^*]+)', response)
#         if glossary_entries:
#             glossary_entries = [f"{term}: {definition}" for term, definition in glossary_entries]
#         else:
#             # Just split by newlines if all else fails
#             glossary_entries = [line for line in response.split('\n') if line.strip()]
    
#     return glossary_entries


# async def generate_section_summaries(transcript_segments, content_analysis):
#     """Generate summaries for major sections of the video."""
#     print("Generating section summaries...")
    
#     # Identify major sections based on natural breaks in content
#     # This is a simplified approach - a more sophisticated approach would 
#     # analyze the transcript for topic shifts
    
#     total_segments = len(transcript_segments)
    
#     # For very short videos, just summarize the whole thing
#     if total_segments < 10:
#         section_texts = [" ".join([seg[2] for seg in transcript_segments])]
#         section_timestamps = ["00:00:00"]
#     else:
#         # Divide into 3-5 sections depending on length
#         num_sections = min(5, max(3, total_segments // 50))
#         section_size = total_segments // num_sections
        
#         section_texts = []
#         section_timestamps = []
        
#         for i in range(num_sections):
#             start_idx = i * section_size
#             end_idx = min((i + 1) * section_size, total_segments)
            
#             section_text = " ".join([seg[2] for seg in transcript_segments[start_idx:end_idx]])
#             section_timestamp = format_timestamp(transcript_segments[start_idx][0])
            
#             section_texts.append(section_text)
#             section_timestamps.append(section_timestamp)
    
#     # Generate summaries for each section
#     section_summaries = []
    
#     for i, (text, timestamp) in enumerate(zip(section_texts, section_timestamps)):
#         if not text.strip():
#             continue
            
#         summary_prompt = f"""
#         Summarize this section of a video transcript in 1-2 paragraphs. Focus on the key points and main ideas.
        
#         TRANSCRIPT SECTION:
#         {text[:3000]}... [truncated if needed]
        
#         Instructions:
#         1. Create a concise summary that captures the essential information in this section
#         2. Include the main ideas, key examples, and important concepts
#         3. Keep the summary to 1-2 paragraphs
#         4. Use clear, straightforward language appropriate for {content_analysis.get('academic_level', 'general')} students
        
#         Return only the summary with no additional text.
#         """
        
#         summary = await generate_response_async(summary_prompt)
        
#         section_summaries.append({
#             "timestamp": timestamp,
#             "title": f"Section {i+1}",  # A more sophisticated approach would generate meaningful titles
#             "summary": summary.strip()
#         })
    
#     return section_summaries


# def format_study_guide_for_display(study_guide):
#     """Format the study guide into a readable text document."""
    
#     title = study_guide["title"]
#     content_analysis = study_guide["content_analysis"]
    
#     formatted_text = f"""# {title}

# ## Overview
# - **Subject:** {content_analysis.get('main_subject', 'Video Content')}
# - **Format:** {content_analysis.get('format', 'Educational Video')}
# - **Level:** {content_analysis.get('academic_level', 'General')}

# ## Learning Objectives
# """

#     # Add learning objectives
#     for i, obj in enumerate(content_analysis.get('learning_objectives', []), 1):
#         formatted_text += f"{i}. {obj}\n"
    
#     # Add section summaries if available
#     if study_guide.get("section_summaries"):
#         formatted_text += "\n## Video Section Summaries\n\n"
#         for section in study_guide["section_summaries"]:
#             formatted_text += f"### {section['title']} (begins at {section['timestamp']})\n\n"
#             formatted_text += f"{section['summary']}\n\n"
    
#     # Add quiz questions
#     formatted_text += "\n## Quiz Questions\n\n"
#     for q in study_guide.get("quiz_questions", []):
#         formatted_text += f"{q}\n\n"
    
#     # Add quiz answer key
#     formatted_text += "\n## Quiz Answer Key\n\n"
#     for a in study_guide.get("quiz_answers", []):
#         formatted_text += f"{a}\n\n"
    
#     # Add essay questions
#     formatted_text += "\n## Essay Questions\n\n"
#     for eq in study_guide.get("essay_questions", []):
#         formatted_text += f"{eq}\n\n"
    
#     # Add glossary
#     formatted_text += "\n## Glossary of Key Terms\n\n"
#     for term in study_guide.get("glossary", []):
#         formatted_text += f"- **{term}**\n"
    
#     # Add footer with source
#     formatted_text += f"\n\n---\n*This study guide was generated based on the video: {study_guide['video_info']['title']}*"
    
#     return formatted_text


# # async def save_study_guide(formatted_study_guide, video_title):
# #     """Save the formatted study guide to a text file."""
# #     try:
# #         # Create output directory if it doesn't exist
# #         output_dir = "study_guides"
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         # Create a file name based on the video title
# #         safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
# #         file_name = f"{safe_title}_study_guide.md"
# #         file_path = os.path.join(output_dir, file_name)
        
# #         # Write to file
# #         with open(file_path, 'w', encoding='utf-8') as f:
# #             f.write(formatted_study_guide)
        
# #         print(f"Study guide saved to: {file_path}")
# #         return file_path
        
# #     except Exception as e:
# #         print(f"Error saving study guide: {e}")
# #         return None

# async def save_study_guide(formatted_study_guide, video_title):
#     """Save the formatted study guide to a text file."""
#     try:
#         # Use OUTPUT_DIR (downloads) defined in main.py
#         output_dir = os.path.join("downloads", "study_guides")  # Ensures study_guides is inside downloads
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Create a file name based on the video title
#         safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
#         file_name = f"{safe_title}_study_guide.md"
#         file_path = os.path.join(output_dir, file_name)
        
#         # Write to file
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(formatted_study_guide)
        
#         print(f"Study guide saved to: {file_path}")
#         return file_path
        
#     except Exception as e:
#         print(f"Error saving study guide: {e}")
#         return None



# async def generate_faq(transcript_segments, video_info):
#     """
#     Generate common questions and answers about the video content.
    
#     Args:
#         transcript_segments: List of (start_time, end_time, text) tuples
#         video_info: Dictionary with video metadata
        
#     Returns:
#         List of question-answer pairs
#     """
#     # Extract full text from transcript
#     full_text = " ".join([seg[2] for seg in transcript_segments])
    
#     # Create a prompt for generating FAQ
#     prompt = f"""
#     Based on this video transcript, generate 5 frequently asked questions (FAQ) that viewers might have.
    
#     VIDEO TITLE: {video_info.get('title', 'Unknown')}
#     VIDEO DESCRIPTION: {video_info.get('description', 'No description')}
    
#     TRANSCRIPT EXCERPT:
#     {full_text[:3000]}... [truncated]
    
#     For each question:
#     1. The question should be specific to the content
#     2. It should target important or interesting information from the video
#     3. Provide a concise, accurate answer based solely on the transcript
    
#     Return the results in this JSON format:
#     {{
#         "faq": [
#             {{
#                 "question": "First question here?",
#                 "answer": "Answer to first question"
#             }},
#             ...
#         ]
#     }}
#     """
    
#     logger.info("Generating FAQ for video content...")
#     response = await generate_response_async(prompt)
#     raw_content = response
    
#     # Extract JSON from response
#     faq_data = None
#     try:
#         # Try to directly parse the response as JSON
#         faq_data = json.loads(raw_content)
#     except json.JSONDecodeError:
#         # Try to extract JSON using regex
#         logger.info("Direct JSON parsing failed, trying regex extraction...")
#         json_match = re.search(r'\{[\s\S]*\}', raw_content)
#         if json_match:
#             try:
#                 faq_data = json.loads(json_match.group(0))
#             except json.JSONDecodeError:
#                 logger.warning("JSON extraction failed")
    
#     # If JSON parsing fails, create a structured response manually
#     if not faq_data or "faq" not in faq_data:
#         logger.warning("Creating structured FAQ manually from text response")
#         faq_list = []
        
#         # Try to extract Q&A pairs from text
#         lines = raw_content.split('\n')
#         current_question = None
#         current_answer = ""
        
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
                
#             # Check if line starts with number followed by question mark
#             q_match = re.match(r'^(\d+[\)\.])?\s*(.+\?)', line)
#             if q_match:
#                 # Save previous Q&A pair if exists
#                 if current_question and current_answer:
#                     faq_list.append({
#                         "question": current_question,
#                         "answer": current_answer.strip()
#                     })
#                     current_answer = ""
                
#                 # New question
#                 current_question = q_match.group(2).strip()
#             elif current_question and not current_answer and ":" in line:
#                 # Handle "Q: question" format
#                 parts = line.split(":", 1)
#                 if parts[0].strip().lower() in ["q", "question"]:
#                     current_question = parts[1].strip()
#                 elif parts[0].strip().lower() in ["a", "answer"]:
#                     current_answer = parts[1].strip()
#             elif current_question:
#                 # Collect answer lines
#                 current_answer += " " + line
        
#         # Add the last Q&A pair
#         if current_question and current_answer:
#             faq_list.append({
#                 "question": current_question,
#                 "answer": current_answer.strip()
#             })
        
#         faq_data = {"faq": faq_list}
    
#     return faq_data.get("faq", [])

import json
import re
import os
import asyncio
from logger_config import logger

# Import your existing dependencies
import ollama
from transcriber import get_youtube_transcript, transcribe_video
from summarizer import generate_response_with_gemini_async, generate_response_with_gemini_async
from utils import format_timestamp


async def generate_study_guide(transcript_segments, video_info):
    """
    Generate a comprehensive study guide for educational content.
    
    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        video_info: Dictionary with video metadata
        
    Returns:
        A structured study guide with questions, answers, and other educational materials
    """
    # Extract full text from transcript
    full_text = " ".join([seg[2] for seg in transcript_segments])
    
    # Step 1: Analyze content to determine if it's educational/instructional
    print("Analyzing video content for educational value...")
    
    content_analysis_prompt = f"""
    Analyze this video transcript to determine if it is educational/instructional content suitable for a study guide.
    
    VIDEO TITLE: {video_info.get('title', 'Unknown')}
    VIDEO DESCRIPTION: {video_info.get('description', 'No description')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:3000]}... [truncated]
    
    Please analyze:
    1. Is this educational content that would benefit from a study guide? (Yes/No)
    2. What is the main subject or topic?
    3. What kind of educational format is it? (lecture, tutorial, talk, presentation, etc.)
    4. What academic level is it appropriate for? (K-12, undergraduate, graduate, professional)
    5. What are 3-5 key learning objectives someone might have from this content?
    
    Return only a JSON object with your analysis:
    {{
        "is_educational": true or false,
        "main_subject": "Subject name",
        "format": "Format type",
        "academic_level": "Level",
        "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"]
    }}
    """
    
    try:
        content_analysis_response = await generate_response_with_gemini_async(content_analysis_prompt)
        
        # Extract JSON data
        json_match = re.search(r'\{[\s\S]*\}', content_analysis_response)
        if json_match:
            content_analysis = json.loads(json_match.group(0))
        else:
            # Default values if JSON extraction fails
            content_analysis = {
                "is_educational": True,  # Assume educational by default
                "main_subject": video_info.get('title', 'Unknown Subject'),
                "format": "Presentation",
                "academic_level": "General",
                "learning_objectives": ["Understanding the main concepts presented in the video"]
            }
            
        # If content is not educational, still generate a basic study guide but note limitations
        if not content_analysis.get("is_educational", True):
            print("Note: This content may not be primarily educational, but creating a study guide anyway.")
    
    except Exception as e:
        print(f"Error analyzing content: {e}")
        # Default values if analysis fails
        content_analysis = {
            "is_educational": True,
            "main_subject": video_info.get('title', 'Unknown Subject'),
            "format": "Presentation",
            "academic_level": "General",
            "learning_objectives": ["Understanding the main concepts presented in the video"]
        }
    
    # Step 2: Generate the study guide sections
    print("Generating comprehensive study guide sections...")
    
    # Generate study guide components in parallel for efficiency
    tasks = [
        generate_quiz_questions(full_text, content_analysis),
        generate_quiz_answers(full_text, content_analysis),
        generate_essay_questions(full_text, content_analysis),
        generate_key_terms_glossary(full_text, content_analysis),
        generate_section_summaries(transcript_segments, content_analysis)
    ]
    
    quiz_questions, quiz_answers, essay_questions, glossary, section_summaries = await asyncio.gather(*tasks)
    
    # Step 3: Generate a title for the study guide
    study_guide_title_prompt = f"""
    Create an engaging, descriptive title for a study guide based on this video content.
    The title should be concise (5-10 words) and clearly indicate the main subject.
    
    VIDEO TITLE: {video_info.get('title', 'Unknown')}
    MAIN SUBJECT: {content_analysis.get('main_subject', 'Unknown')}
    LEARNING OBJECTIVES: {', '.join(content_analysis.get('learning_objectives', ['Understanding the content']))}
    
    Return only the title with no other text.
    """
    
    study_guide_title = await generate_response_with_gemini_async(study_guide_title_prompt)
    study_guide_title = study_guide_title.strip().replace('"', '').replace("'", "")
    if ":" not in study_guide_title:
        study_guide_title += f": A Study Guide Based on {video_info.get('title', 'the Video')}"
    
    # Step 4: Assemble the complete study guide
    study_guide = {
        "title": study_guide_title,
        "video_info": {
            "title": video_info.get('title', 'Unknown'),
            "description": video_info.get('description', 'No description')
        },
        "content_analysis": content_analysis,
        "quiz_questions": quiz_questions,
        "quiz_answers": quiz_answers,
        "essay_questions": essay_questions,
        "glossary": glossary,
        "section_summaries": section_summaries
    }
    
    # Step 5: Format the study guide for display
    formatted_study_guide = format_study_guide_for_display(study_guide)
    
    # Step 6: Save the study guide to a file
    file_path = await save_study_guide(formatted_study_guide, video_info.get('title', 'study_guide'))
    
    return {
        "study_guide": study_guide,
        "formatted_study_guide": formatted_study_guide,
        "file_path": file_path
    }


async def generate_quiz_questions(full_text, content_analysis):
    """Generate a set of quiz questions based on the video content."""
    print("Generating quiz questions...")
    
    quiz_prompt = f"""
    You are an experienced educator. Create 10 well-crafted quiz questions based on this video transcript.
    
    SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
    ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:4000]}... [truncated]
    
    Instructions:
    1. Create questions that assess understanding of key concepts, not just recall of minor details
    2. Include a mix of question types (factual recall, concept application, analysis)
    3. Focus on the most important points from the video
    4. Ensure questions are clear, unambiguous, and answerable from the transcript
    5. Phrase questions in a direct, concise manner
    6. Number each question from 1-10
    
    Return ONLY the questions without answers or explanations, one per line, numbered 1-10.
    """
    
    response = await generate_response_with_gemini_async(quiz_prompt)
    
    # Clean up the response
    questions = []
    current_question = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number (1-10)
        if re.match(r'^\d{1,2}[\.\)]', line):
            # If we have a current question, save it
            if current_question:
                questions.append(current_question)
            # Start a new question
            current_question = line
        else:
            # Continue the current question
            current_question += " " + line
    
    # Add the last question
    if current_question:
        questions.append(current_question)
    
    # Ensure we have questions (if parsing failed, use the whole response)
    if not questions:
        questions = [response]
    
    return questions


async def generate_quiz_answers(full_text, content_analysis):
    """Generate answers for the quiz questions."""
    print("Generating quiz answers...")
    
    answers_prompt = f"""
    You are an experienced educator. Create a comprehensive answer key for a 10-question quiz about this video.
    
    SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
    ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:4000]}... [truncated]
    
    Instructions:
    1. First, generate 10 likely questions that would appear on a quiz about this content
    2. Then, provide detailed answers (3-5 sentences each) that thoroughly explain the correct response
    3. Include relevant examples or supporting details from the transcript
    4. Ensure answers are comprehensive enough to serve as a study guide
    5. Number each answer from 1-10 to match the questions
    
    Format each answer as "X. [Detailed answer explanation]" where X is the question number.
    Return ONLY the answers, one per numbered point.
    """
    
    response = await generate_response_with_gemini_async(answers_prompt)
    
    # Parse answers with the same approach as questions
    answers = []
    current_answer = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number (1-10)
        if re.match(r'^\d{1,2}[\.\)]', line):
            # If we have a current answer, save it
            if current_answer:
                answers.append(current_answer)
            # Start a new answer
            current_answer = line
        else:
            # Continue the current answer
            current_answer += " " + line
    
    # Add the last answer
    if current_answer:
        answers.append(current_answer)
    
    # Ensure we have answers (if parsing failed, use the whole response)
    if not answers:
        answers = [response]
    
    return answers


async def generate_essay_questions(full_text, content_analysis):
    """Generate thought-provoking essay questions based on the content."""
    print("Generating essay questions...")
    
    essay_prompt = f"""
    You are an experienced educator. Create 5 thought-provoking essay questions based on this video transcript.
    
    SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
    ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    LEARNING OBJECTIVES: {', '.join(content_analysis.get('learning_objectives', ['Understanding the content']))}
    
    TRANSCRIPT EXCERPT:
    {full_text[:4000]}... [truncated]
    
    Instructions:
    1. Create essay questions that require critical thinking, analysis, evaluation, or synthesis
    2. Questions should prompt in-depth exploration of the main concepts from the video
    3. Each question should be suitable for a 1-2 page response
    4. Focus on controversial, nuanced, or complex aspects of the topic
    5. Each question should open with a relevant action verb (Analyze, Evaluate, Compare, etc.)
    
    Return only the 5 essay questions, one per line. Number them 1-5.
    """
    
    response = await generate_response_with_gemini_async(essay_prompt)
    
    # Parse the essay questions
    essay_questions = []
    current_question = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number (1-5)
        if re.match(r'^\d{1}[\.\)]', line):
            # If we have a current question, save it
            if current_question:
                essay_questions.append(current_question)
            # Start a new question
            current_question = line
        else:
            # Continue the current question
            current_question += " " + line
    
    # Add the last question
    if current_question:
        essay_questions.append(current_question)
    
    # Ensure we have questions (if parsing failed, use the whole response)
    if not essay_questions:
        essay_questions = [response]
    
    return essay_questions


async def generate_key_terms_glossary(full_text, content_analysis):
    """Generate a glossary of key terms from the video content."""
    print("Generating key terms glossary...")
    
    glossary_prompt = f"""
    You are an educational content creator. Create a glossary of 8-12 important key terms used in this video transcript.
    
    SUBJECT: {content_analysis.get('main_subject', 'the video topic')}
    ACADEMIC LEVEL: {content_analysis.get('academic_level', 'General')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:4000]}... [truncated]
    
    Instructions:
    1. Identify specialized terminology, jargon, or important concepts mentioned in the video
    2. For each term, provide a clear, concise definition (1-2 sentences) based on how it was used in the video
    3. Focus on terms that are central to understanding the main concepts
    4. Format as "Term: Definition"
    5. Include only terms that actually appear in the transcript
    
    Return ONLY the glossary entries, one term per line, in alphabetical order.
    """
    
    response = await generate_response_with_gemini_async(glossary_prompt)
    
    # Parse the glossary entries
    glossary_entries = []
    current_entry = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains a term definition (Term: Definition)
        if re.match(r'^[A-Za-z\s\(\)]+:', line):
            # If we have a current entry, save it
            if current_entry:
                glossary_entries.append(current_entry)
            # Start a new entry
            current_entry = line
        else:
            # Continue the current entry
            current_entry += " " + line
    
    # Add the last entry
    if current_entry:
        glossary_entries.append(current_entry)
    
    # Ensure we have entries (if parsing failed, try alternative parsing)
    if not glossary_entries:
        # Try alternative parsing - look for bold/emphasized terms
        glossary_entries = re.findall(r'\*\*([^*]+)\*\*:([^*]+)', response)
        if glossary_entries:
            glossary_entries = [f"{term}: {definition}" for term, definition in glossary_entries]
        else:
            # Just split by newlines if all else fails
            glossary_entries = [line for line in response.split('\n') if line.strip()]
    
    return glossary_entries


async def generate_section_summaries(transcript_segments, content_analysis):
    """Generate summaries for major sections of the video."""
    print("Generating section summaries...")
    
    # Identify major sections based on natural breaks in content
    # This is a simplified approach - a more sophisticated approach would 
    # analyze the transcript for topic shifts
    
    total_segments = len(transcript_segments)
    
    # For very short videos, just summarize the whole thing
    if total_segments < 10:
        section_texts = [" ".join([seg[2] for seg in transcript_segments])]
        section_timestamps = ["00:00:00"]
    else:
        # Divide into 3-5 sections depending on length
        num_sections = min(5, max(3, total_segments // 50))
        section_size = total_segments // num_sections
        
        section_texts = []
        section_timestamps = []
        
        for i in range(num_sections):
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, total_segments)
            
            section_text = " ".join([seg[2] for seg in transcript_segments[start_idx:end_idx]])
            section_timestamp = format_timestamp(transcript_segments[start_idx][0])
            
            section_texts.append(section_text)
            section_timestamps.append(section_timestamp)
    
    # Generate summaries for each section
    section_summaries = []
    
    for i, (text, timestamp) in enumerate(zip(section_texts, section_timestamps)):
        if not text.strip():
            continue
            
        summary_prompt = f"""
        Summarize this section of a video transcript in 1-2 paragraphs. Focus on the key points and main ideas.
        
        TRANSCRIPT SECTION:
        {text[:3000]}... [truncated if needed]
        
        Instructions:
        1. Create a concise summary that captures the essential information in this section
        2. Include the main ideas, key examples, and important concepts
        3. Keep the summary to 1-2 paragraphs
        4. Use clear, straightforward language appropriate for {content_analysis.get('academic_level', 'general')} students
        
        Return only the summary with no additional text.
        """
        
        summary = await generate_response_with_gemini_async(summary_prompt)
        
        section_summaries.append({
            "timestamp": timestamp,
            "title": f"Section {i+1}",  # A more sophisticated approach would generate meaningful titles
            "summary": summary.strip()
        })
    
    return section_summaries


def format_study_guide_for_display(study_guide):
    """Format the study guide into a readable text document."""
    
    title = study_guide["title"]
    content_analysis = study_guide["content_analysis"]
    
    formatted_text = f"""# {title}

## Overview
- **Subject:** {content_analysis.get('main_subject', 'Video Content')}
- **Format:** {content_analysis.get('format', 'Educational Video')}
- **Level:** {content_analysis.get('academic_level', 'General')}

## Learning Objectives
"""

    # Add learning objectives
    for i, obj in enumerate(content_analysis.get('learning_objectives', []), 1):
        formatted_text += f"{i}. {obj}\n"
    
    # Add section summaries if available
    if study_guide.get("section_summaries"):
        formatted_text += "\n## Video Section Summaries\n\n"
        for section in study_guide["section_summaries"]:
            formatted_text += f"### {section['title']} (begins at {section['timestamp']})\n\n"
            formatted_text += f"{section['summary']}\n\n"
    
    # Add quiz questions
    formatted_text += "\n## Quiz Questions\n\n"
    for q in study_guide.get("quiz_questions", []):
        formatted_text += f"{q}\n\n"
    
    # Add quiz answer key
    formatted_text += "\n## Quiz Answer Key\n\n"
    for a in study_guide.get("quiz_answers", []):
        formatted_text += f"{a}\n\n"
    
    # Add essay questions
    formatted_text += "\n## Essay Questions\n\n"
    for eq in study_guide.get("essay_questions", []):
        formatted_text += f"{eq}\n\n"
    
    # Add glossary
    formatted_text += "\n## Glossary of Key Terms\n\n"
    for term in study_guide.get("glossary", []):
        formatted_text += f"- **{term}**\n"
    
    # Add footer with source
    formatted_text += f"\n\n---\n*This study guide was generated based on the video: {study_guide['video_info']['title']}*"
    
    return formatted_text


async def save_study_guide(formatted_study_guide, video_title):
    """Save the formatted study guide to a text file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = "study_guides"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a file name based on the video title
        safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
        file_name = f"{safe_title}_study_guide.txt"
        file_path = os.path.join(output_dir, file_name)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_study_guide)
        
        print(f"Study guide saved to: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error saving study guide: {e}")
        return None


async def generate_faq(transcript_segments, video_info):
    """
    Generate common questions and answers about the video content.
    
    Args:
        transcript_segments: List of (start_time, end_time, text) tuples
        video_info: Dictionary with video metadata
        
    Returns:
        List of question-answer pairs
    """
    # Extract full text from transcript
    full_text = " ".join([seg[2] for seg in transcript_segments])
    
    # Create a prompt for generating FAQ
    prompt = f"""
    Based on this video transcript, generate 5 frequently asked questions (FAQ) that viewers might have.
    
    VIDEO TITLE: {video_info.get('title', 'Unknown')}
    VIDEO DESCRIPTION: {video_info.get('description', 'No description')}
    
    TRANSCRIPT EXCERPT:
    {full_text[:3000]}... [truncated]
    
    For each question:
    1. The question should be specific to the content
    2. It should target important or interesting information from the video
    3. Provide a concise, accurate answer based solely on the transcript
    
    Return the results in this JSON format:
    {{
        "faq": [
            {{
                "question": "First question here?",
                "answer": "Answer to first question"
            }},
            ...
        ]
    }}
    """
    
    logger.info("Generating FAQ for video content...")
    response = await generate_response_with_gemini_async(prompt)
    raw_content = response
    
    # Extract JSON from response
    faq_data = None
    try:
        # Try to directly parse the response as JSON
        faq_data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON using regex
        logger.info("Direct JSON parsing failed, trying regex extraction...")
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            try:
                faq_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("JSON extraction failed")
    
    # If JSON parsing fails, create a structured response manually
    if not faq_data or "faq" not in faq_data:
        logger.warning("Creating structured FAQ manually from text response")
        faq_list = []
        
        # Try to extract Q&A pairs from text
        lines = raw_content.split('\n')
        current_question = None
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with number followed by question mark
            q_match = re.match(r'^(\d+[\)\.])?\s*(.+\?)', line)
            if q_match:
                # Save previous Q&A pair if exists
                if current_question and current_answer:
                    faq_list.append({
                        "question": current_question,
                        "answer": current_answer.strip()
                    })
                    current_answer = ""
                
                # New question
                current_question = q_match.group(2).strip()
            elif current_question and not current_answer and ":" in line:
                # Handle "Q: question" format
                parts = line.split(":", 1)
                if parts[0].strip().lower() in ["q", "question"]:
                    current_question = parts[1].strip()
                elif parts[0].strip().lower() in ["a", "answer"]:
                    current_answer = parts[1].strip()
            elif current_question:
                # Collect answer lines
                current_answer += " " + line
        
        # Add the last Q&A pair
        if current_question and current_answer:
            faq_list.append({
                "question": current_question,
                "answer": current_answer.strip()
            })
        
        faq_data = {"faq": faq_list}
    
    return faq_data.get("faq", [])