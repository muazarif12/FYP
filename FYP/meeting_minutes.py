# import time
# import re
# import ollama
# import asyncio
# import os
# import json
# from constants import OUTPUT_DIR
# from logger_config import logger
# from utils import format_timestamp

# async def detect_meeting_type(transcript_text):
#     """
#     Detect the type of meeting based on transcript content.
    
#     Args:
#         transcript_text: Full transcript text
        
#     Returns:
#         Tuple of (meeting_type, confidence) where meeting_type is a string
#         and confidence is a float between 0 and 1
#     """
#     # Define meeting type keywords
#     meeting_types = {
#         "standup": ["standup", "stand-up", "daily", "yesterday", "today", "blockers", "impediments", "scrum"],
#         "planning": ["sprint planning", "planning", "backlog", "user stories", "story points", "capacity", "roadmap"],
#         "retrospective": ["retro", "retrospective", "went well", "improvement", "action items", "went wrong"],
#         "review": ["review", "demo", "demonstration", "sprint review", "showcase", "progress", "key review", "engineering review"],
#         "board": ["board meeting", "shareholders", "quarterly", "financials", "budget", "governance"],
#         "interview": ["interview", "candidate", "resume", "skills", "experience", "hire", "position"],
#         "team": ["team meeting", "all-hands", "status update", "department", "organization"],
#         "client": ["client", "customer", "requirements", "feedback", "proposal", "project status"]
#     }
    
#     # Count occurrences of keywords
#     scores = {mtype: 0 for mtype in meeting_types}
    
#     # Normalize text for better matching
#     text_lower = transcript_text.lower()
    
#     # Calculate scores
#     for mtype, keywords in meeting_types.items():
#         for keyword in keywords:
#             count = text_lower.count(keyword)
#             scores[mtype] += count
    
#     # Find max score
#     max_score = max(scores.values()) if scores.values() else 0
    
#     # If no significant pattern, default to general meeting
#     if max_score < 3:
#         return "general", 0.5
    
#     # Get the meeting type with highest score
#     best_type = max(scores.items(), key=lambda x: x[1])[0]
    
#     # Calculate confidence (normalized score)
#     total = sum(scores.values())
#     confidence = scores[best_type] / total if total > 0 else 0.5
    
#     return best_type, confidence


# # Add this new function to meeting_minutes.py to handle timestamped transcripts

# async def process_timestamped_transcript(transcript_text):
#     """
#     Process timestamped transcript to extract cleaner content for analysis.
    
#     Args:
#         transcript_text: Raw timestamped transcript
        
#     Returns:
#         Cleaner transcript text with speakers identified where possible
#     """
#     processed_text = ""
#     current_speaker = None
#     speaker_pattern = re.compile(r'(\w+)\s*:')
#     timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:')
    
#     # Split by lines
#     lines = transcript_text.split('\n')
#     for line in lines:
#         # Skip empty lines
#         if not line.strip():
#             continue
            
#         # Check if this is a timestamp line
#         if timestamp_pattern.match(line):
#             continue
            
#         # Try to detect speaker
#         speaker_match = speaker_pattern.search(line)
#         if speaker_match:
#             potential_speaker = speaker_match.group(1)
#             # Verify this looks like a name (not too short, not a number)
#             if len(potential_speaker) > 2 and not re.match(r'\d', potential_speaker):
#                 current_speaker = potential_speaker
#                 # Add the speaker prefix
#                 speaker_text = line.split(':', 1)
#                 if len(speaker_text) > 1:
#                     processed_text += f"{current_speaker}: {speaker_text[1].strip()}\n"
#                 else:
#                     processed_text += f"{current_speaker}: {line.strip()}\n"
#             else:
#                 # Add line without speaker
#                 processed_text += f"{line.strip()}\n"
#         else:
#             # Add line with current speaker if available
#             if current_speaker:
#                 processed_text += f"{current_speaker} (cont'd): {line.strip()}\n"
#             else:
#                 processed_text += f"{line.strip()}\n"
    
#     return processed_text

# async def generate_meeting_minutes(transcript_segments, video_info, detected_language="en", timestamped_transcript=None):
#     """
#     Generate structured meeting minutes from video transcript.
    
#     Args:
#         transcript_segments: List of transcript segments (start_time, end_time, text)
#         video_info: Dictionary with video metadata
#         detected_language: Language code for the transcript
#         timestamped_transcript: Optional timestamped transcript for better processing
        
#     Returns:
#         Dictionary containing structured meeting minutes
#     """
#     start_time = time.time()
    
#     # Extract text from transcript segments
#     full_text = " ".join([seg[2] for seg in transcript_segments])
    
#     # Process timestamped transcript if available
#     if timestamped_transcript:
#         processed_transcript = await process_timestamped_transcript(timestamped_transcript)
#         # Use both for better context
#         combined_text = f"TIMESTAMPED TRANSCRIPT:\n{processed_transcript}\n\nFULL TEXT:\n{full_text}"
#     else:
#         combined_text = full_text
    
#     # Truncate very long transcripts to avoid token limits
#     max_length = 12000  # Characters
#     if len(combined_text) > max_length:
#         logger.info(f"Transcript too long ({len(combined_text)} chars), truncating to {max_length} chars")
#         if timestamped_transcript:
#             # Keep more of the timestamped portion as it's more structured
#             third = max_length // 3
#             processed_part = processed_transcript[:third*2]
#             full_part = full_text[:third]
#             combined_text = f"TIMESTAMPED TRANSCRIPT:\n{processed_part}\n\nFULL TEXT:\n{full_part}"
#         else:
#             # Regular truncation strategy
#             third = max_length // 3
#             combined_text = combined_text[:third] + "..." + combined_text[len(combined_text)//2-third//2:len(combined_text)//2+third//2] + "..." + combined_text[-third:]
    
#     # Extract meeting date from transcript if possible
#     date_pattern = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}'
#     date_matches = re.findall(date_pattern, combined_text.lower())
#     meeting_date = date_matches[0] if date_matches else "Date not specified"
    
#     # Detect meeting type
#     meeting_type, confidence = await detect_meeting_type(combined_text)
#     logger.info(f"Detected meeting type: {meeting_type} (confidence: {confidence:.2f})")
    
#     # Extract participants more effectively
#     participants = extract_participants(combined_text)
    
#     # Determine meeting information
#     video_title = video_info.get('title', 'Unknown Meeting')
#     video_duration = transcript_segments[-1][1] if transcript_segments else 600  # Default to 10 minutes
    
#     # Create meeting prompt based on meeting type
#     if meeting_type == "review":
#         # Enhanced prompt for review meetings with more specific guidance
#         prompt = f"""
#         You are an expert meeting minutes generator specializing in engineering and business review meetings.
#         Given the transcript of a meeting, create professional, structured meeting minutes that capture all essential information.
        
#         MEETING DETAILS:
#         - Title: "{video_title}"
#         - Date: {meeting_date}
#         - Duration: {video_duration:.1f} seconds
#         - Meeting Type: {meeting_type.capitalize()} Meeting
        
#         PARTICIPANTS IDENTIFIED:
#         {', '.join(participants) if participants else "Participants not explicitly identified"}
        
#         VERY IMPORTANT INSTRUCTIONS:
#         1. EXTRACT ALL PARTICIPANTS: Identify every person mentioned in the transcript, especially speakers.
#            Look for names followed by speaking turns, references to individuals by name, and people assigned tasks.
        
#         2. IDENTIFY EVERY AGENDA ITEM: The meeting likely discusses multiple topics.
#            Look for: 
#            - Numbered agenda items mentioned explicitly
#            - Topic introductions or transitions
#            - Distinct discussion segments on different subjects
#            - Questions or issues raised for discussion
        
#         3. CAPTURE ALL METRICS AND KPIs: Identify metrics discussed including:
#            - Performance indicators with specific values
#            - Trends and comparisons to previous periods
#            - Status updates (below/above target, improving/declining)
#            - Any numerical measurements of business or technical performance
        
#         4. RECORD ALL DECISIONS: Capture every decision made, including:
#            - Explicit statements of decisions ("we decided to...")
#            - Agreements reached after discussion
#            - Approvals or rejections of proposals
#            - Consensus statements on future direction
        
#         5. IDENTIFY ALL ACTION ITEMS: Find every task assignment:
#            - Who is assigned to do what
#            - Due dates or timeframes if specified
#            - The context and rationale for each task
#            - Follow-up actions mentioned
        
#         TRANSCRIPT CONTENT:
#         {combined_text}
        
#         Generate comprehensive meeting minutes in this JSON structure:
#         {{
#           "title": "Meeting Minutes: {video_title}",
#           "date": "{meeting_date}",
#           "duration": "{int(video_duration / 60)} minutes",
#           "participants": [
#             "List all participant names from the transcript - be thorough"
#           ],
#           "agenda_items": [
#             {{
#               "topic": "First agenda item with item number if available",
#               "discussion": "Detailed summary of what was discussed on this topic",
#               "speakers": ["People who spoke on this topic"]
#             }},
#             {{
#               "topic": "Second agenda item",
#               "discussion": "Detailed summary of what was discussed",
#               "speakers": ["People who spoke on this topic"]
#             }}
#           ],
#           "key_metrics_discussed": [
#             {{
#               "metric": "Name of the metric or KPI discussed",
#               "status": "Current status as mentioned in the meeting",
#               "details": "Comprehensive details including numbers, trends, and context"
#             }}
#           ],
#           "action_items": [
#             {{
#               "task": "Specific task to be done",
#               "assigned_to": "Person assigned to the task",
#               "due_date": "Due date if mentioned or 'Not specified'",
#               "context": "Why this task is needed and any relevant details"
#             }}
#           ],
#           "decisions": [
#             "First decision made in clear, specific language",
#             "Second decision made in clear, specific language"
#           ],
#           "next_steps": [
#             "First follow-up action with any assigned owner",
#             "Second follow-up action with any assigned owner"
#           ]
#         }}
#         """
#     else:
#         # Enhanced prompt for other meeting types
#         prompt = f"""
#         You are an expert meeting minutes generator. Given the transcript of a meeting, create professional, 
#         structured meeting minutes that capture all essential information.
        
#         MEETING DETAILS:
#         - Title: "{video_title}"
#         - Date: {meeting_date}
#         - Duration: {video_duration:.1f} seconds
#         - Meeting Type: {meeting_type.capitalize()} Meeting
        
#         PARTICIPANTS IDENTIFIED:
#         {', '.join(participants) if participants else "Participants not explicitly identified"}
        
#         VERY IMPORTANT INSTRUCTIONS:
#         1. EXTRACT ALL PARTICIPANTS: Identify every person mentioned in the transcript.
#            Look for names of speakers and references to individuals.
        
#         2. IDENTIFY MAIN TOPICS: The meeting likely covers several important topics.
#            Look for topic shifts, discussion themes, and key conversation points.
        
#         3. CAPTURE KEY POINTS: For each topic, identify the most important information:
#            - Main statements and arguments
#            - Supporting details and examples
#            - Questions raised and responses given
#            - Important announcements or updates
        
#         4. RECORD ALL DECISIONS: Capture every decision mentioned:
#            - Explicit decisions made during the meeting
#            - Agreements reached after discussion
#            - Conclusions on topics or issues
        
#         5. IDENTIFY ALL ACTION ITEMS: Find all tasks assigned:
#            - Who is responsible for what task
#            - Any deadlines mentioned
#            - The purpose of each task
        
#         TRANSCRIPT CONTENT:
#         {combined_text}
        
#         Generate comprehensive meeting minutes following this exact JSON structure:
#         {{
#           "title": "Meeting Minutes: {video_title}",
#           "date": "{meeting_date}",
#           "duration": "{int(video_duration / 60)} minutes",
#           "participants": [
#             "List all participants mentioned in the transcript - be thorough"
#           ],
#           "agenda": [
#             "First main topic discussed in the meeting",
#             "Second main topic discussed in the meeting"
#           ],
#           "key_points": [
#             {{
#               "topic": "First important topic",
#               "points": [
#                 "First key point about this topic",
#                 "Second key point about this topic",
#                 "Third key point about this topic"
#               ],
#               "timestamp": "Approximate timestamp if available"
#             }},
#             {{
#               "topic": "Second important topic",
#               "points": [
#                 "First key point about this topic",
#                 "Second key point about this topic"
#               ],
#               "timestamp": "Approximate timestamp if available"
#             }}
#           ],
#           "action_items": [
#             {{
#               "task": "Specific task to be done",
#               "assigned_to": "Person/team assigned",
#               "due_date": "Due date if mentioned, otherwise 'Not specified'",
#               "notes": "Any additional context or information"
#             }}
#           ],
#           "decisions": [
#             "First decision made during the meeting",
#             "Second decision made during the meeting"
#           ],
#           "next_meeting": "Details about the next meeting if mentioned, otherwise 'Not specified'"
#         }}
#         """
    
#     # Add JSON output instruction to ensure proper format
#     prompt += """
    
#     IMPORTANT GUIDELINES:
#     1. Focus on accuracy - only include information from the transcript
#     2. Format your response as VALID JSON only - this is critical
#     3. Do NOT use "//", "..." or any comments in the JSON
#     4. Use empty arrays [] for sections with no content
#     5. Return only the JSON object, nothing else
#     6. Be thorough and comprehensive - capture all important meeting content
#     7. Make sure to extract specific, detailed information rather than generic statements
#     """
    
#     try:
#         logger.info("Generating meeting minutes with LLM...")
        
#         # Run Ollama model with explicit instruction for JSON output
#         response = ollama.chat(
#             model="deepseek-r1:7b",
#             messages=[
#                 {"role": "system", "content": "You are a meeting summarization assistant. You create detailed, comprehensive meeting minutes in JSON format."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
        
#         end_time = time.time()
#         logger.info(f"Time taken to generate meeting minutes: {end_time - start_time:.4f} seconds")
        
#         raw_content = response["message"]["content"]
        
#         # Add a simpler JSON extraction method first - look for content between braces
#         minutes_data = extract_json(raw_content)
        
#         # If JSON parsing fails, use the fallback structure
#         if not minutes_data:
#             logger.warning("Failed to parse meeting minutes as JSON, creating basic structure")
            
#             # Create an improved fallback structure with some extracted info
#             if meeting_type == "review":
#                 minutes_data = {
#                     "title": f"Meeting Minutes: {video_title}",
#                     "date": meeting_date,
#                     "duration": f"{int(video_duration / 60)} minutes",
#                     "participants": participants if participants else ["Participants not explicitly identified"],
#                     "agenda_items": [
#                         {
#                             "topic": "Meeting Overview",
#                             "discussion": "Review of key metrics, issues, and proposals",
#                             "speakers": extract_main_speakers(combined_text) or ["Various participants"]
#                         }
#                     ],
#                     "key_metrics_discussed": extract_metrics(combined_text),
#                     "action_items": extract_action_items(combined_text),
#                     "decisions": extract_decisions(combined_text) or ["No clear decisions identified"],
#                     "next_steps": ["Follow up on discussed items"]
#                 }
#             else:
#                 minutes_data = {
#                     "title": f"Meeting Minutes: {video_title}",
#                     "date": meeting_date,
#                     "duration": f"{int(video_duration / 60)} minutes",
#                     "participants": participants if participants else ["Participants not explicitly identified"],
#                     "agenda": ["Topics discussed in meeting"],
#                     "key_points": [{"topic": "General Discussion", "points": ["See transcript for details"], "timestamp": "00:00:00"}],
#                     "action_items": extract_action_items(combined_text),
#                     "decisions": extract_decisions(combined_text) or ["No clear decisions identified"],
#                     "next_meeting": "Not specified"
#                 }
            
#         return minutes_data
        
#     except Exception as e:
#         logger.error(f"Error generating meeting minutes: {e}")
#         # Return basic structure in case of error
#         return {
#             "title": f"Meeting Minutes: {video_title}",
#             "error": f"Failed to generate meeting minutes: {str(e)}",
#             "duration": f"{int(video_duration / 60)} minutes",
#             "participants": participants if participants else ["Error processing participants"],
#             "key_points": [{"topic": "Error", "points": ["Error processing transcript"], "timestamp": "00:00:00"}]
#         }

# def extract_json(content):
#     """Improved JSON extraction from LLM output"""
#     try:
#         # First try direct loading - might work if content is clean
#         return json.loads(content)
#     except:
#         pass
    
#     # Try to extract JSON content between outermost braces
#     try:
#         # Find the first opening brace and last closing brace
#         start_idx = content.find('{')
#         end_idx = content.rfind('}')
        
#         if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
#             json_str = content[start_idx:end_idx+1]
            
#             # Clean up JSON string
#             # Replace single quotes with double quotes
#             json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
#             # Remove comments
#             json_str = re.sub(r'//.*', '', json_str)
#             # Remove trailing commas in arrays and objects
#             json_str = re.sub(r',\s*}', '}', json_str)
#             json_str = re.sub(r',\s*]', ']', json_str)
            
#             # Try to load the cleaned string
#             try:
#                 return json.loads(json_str)
#             except:
#                 # If still failing, try a more aggressive cleanup
#                 # Replace any non-standard JSON formatting
#                 json_str = re.sub(r'[\n\r\t]', ' ', json_str)
#                 json_str = re.sub(r'\s+', ' ', json_str)
#                 # Try once more with clean string
#                 return json.loads(json_str)
#     except:
#         pass
    
#     # If all attempts fail, return None
#     return None


    
# def extract_participants(text):
#     """Extract likely participant names from the meeting transcript"""
#     # Look for common patterns indicating speakers
#     speaker_patterns = [
#         r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?):',  # Name: pattern
#         r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?) says',  # Name says pattern
#         r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?) (?:mentioned|noted|asked|suggested|proposed)'  # Name action patterns
#     ]
    
#     # Initialize a set to collect participants without duplicates
#     participants_set = set()
    
#     for pattern in speaker_patterns:
#         matches = re.findall(pattern, text)
#         for m in matches:
#             if len(m.strip()) > 2:
#                 participants_set.add(m.strip())
    
#     # Look specifically for names mentioned alongside roles
#     role_patterns = [
#         r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?),?\s+(?:the|our)?\s*(?:CEO|CTO|CFO|COO|manager|director|lead|head)'
#     ]
    
#     for pattern in role_patterns:
#         matches = re.findall(pattern, text)
#         for m in matches:
#             if len(m.strip()) > 2:
#                 participants_set.add(m.strip())
    
#     # Filter out common false positives
#     false_positives = {'The', 'This', 'That', 'These', 'Those', 'They', 'Then', 'There'}
    
#     # Extract names like "John", "Tom", "Mary" mentioned in the text
#     # First after punctuation (fixed width lookbehind)
#     for punct in ['.', ',', ';', ':', '!', '?']:
#         pattern = r'{}\s+([A-Z][a-z]{{2,}})\b'.format(re.escape(punct))
#         name_candidates = re.findall(pattern, text)
#         for name in name_candidates:
#             if name not in false_positives and len(name) > 2:
#                 participants_set.add(name)
    
#     # After line breaks or multiple spaces (no lookbehind)
#     line_start_pattern = r'(?:^|\n)\s*([A-Z][a-z]{2,})\b'
#     name_candidates = re.findall(line_start_pattern, text)
#     for name in name_candidates:
#         if name not in false_positives and len(name) > 2:
#             participants_set.add(name)
    
#     space_pattern = r'\s{2,}([A-Z][a-z]{2,})\b'
#     name_candidates = re.findall(space_pattern, text)
#     for name in name_candidates:
#         if name not in false_positives and len(name) > 2:
#             participants_set.add(name)
    
#     # Extract likely names mentioned in "hi this is [name]" pattern
#     intro_matches = re.findall(r'(?:hi|hello|hey)(?:\s+\w+){0,3}\s+(?:this\s+is\s+|is\s+|am\s+)(\w+(?:\s+\w+)?)', text.lower())
#     for match in intro_matches:
#         name_parts = match.split()
#         if name_parts:
#             name = ' '.join(part.capitalize() for part in name_parts)
#             if len(name) > 2 and name not in false_positives:
#                 participants_set.add(name)
    
#     # Convert set to list before returning
#     return list(participants_set)

# def extract_main_speakers(text):
#     """Extract the most frequent speakers from the transcript"""
#     speaker_pattern = re.compile(r'(\w+)\s*:')
#     matches = speaker_pattern.findall(text)
    
#     # Count occurrences
#     speaker_counts = {}
#     for speaker in matches:
#         if len(speaker) > 2:  # Filter out very short matches
#             speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
#     # Get top speakers (those mentioned more than once)
#     top_speakers = [speaker for speaker, count in speaker_counts.items() if count > 1]
    
#     return top_speakers[:5] if top_speakers else []  # Return up to 5 main speakers

# def extract_metrics(text):
#     """Extract metrics mentioned in the transcript"""
#     # Look for common metric patterns with simpler regex
#     metric_patterns = [
#         r'(\w+(?:\s+\w+){0,3})\s+rate\s+(?:is|was|at)\s+([\d.]+|below target|above target|on target)',
#         r'(\w+(?:\s+\w+){0,3})\s+score\s+(?:is|was|at)\s+([\d.]+|below target|above target|on target)',
#         r'(\w+(?:\s+\w+){0,3})\s+metric\s+(?:is|was|at)\s+([\d.]+|below target|above target|on target)',
#         r'(\w+(?:\s+\w+){0,3})\s+kpi\s+(?:is|was|at)\s+([\d.]+|below target|above target|on target)',
#     ]
    
#     metrics = []
    
#     for pattern in metric_patterns:
#         matches = re.findall(pattern, text.lower())
#         for match in matches:
#             metric_name = match[0].strip().upper()
#             status = match[1].strip()
#             metrics.append({
#                 "metric": metric_name,
#                 "status": status,
#                 "details": f"Mentioned in transcript: {metric_name} is {status}"
#             })
    
#     # Look specifically for MR rate mentions
#     mr_matches = re.findall(r'(mr\s+rate|merge\s+request\s+rate)', text.lower())
#     if mr_matches:
#         metrics.append({
#             "metric": "MR Rate",
#             "status": "Discussed",
#             "details": "Merge Request rate was discussed in the meeting"
#         })
    
#     # Look for SUS mentions
#     sus_matches = re.findall(r'\bsus\b', text.lower())
#     if sus_matches:
#         metrics.append({
#             "metric": "SUS",
#             "status": "Discussed",
#             "details": "System Usability Scale (SUS) was discussed"
#         })
    
#     return metrics if metrics else [{
#         "metric": "Meeting Metrics",
#         "status": "Reviewed",
#         "details": "Various metrics were discussed but specific details not extracted"
#     }]

# def extract_action_items(text):
#     """Extract likely action items from the transcript"""
#     # Look for common action item patterns with simpler regex
#     action_patterns = [
#         r'(\w+)\s+will\s+([\w\s]+)',
#         r'(\w+)\s+should\s+([\w\s]+)',
#         r'(\w+)\s+needs to\s+([\w\s]+)',
#         r'(\w+)\s+going to\s+([\w\s]+)',
#         r'action item for\s+(\w+):\s+([\w\s]+)',
#         r'task for\s+(\w+):\s+([\w\s]+)',
#         r'(\w+)\s+is responsible for\s+([\w\s]+)',
#     ]
    
#     action_items = []
    
#     for pattern in action_patterns:
#         matches = re.findall(pattern, text, re.IGNORECASE)
#         for match in matches:
#             person = match[0].strip()
#             task = match[1].strip()
            
#             # Filter out false positives
#             if len(person) > 2 and len(task) > 5 and not re.match(r'\b(it|that|this|we|they|you|there)\b', person.lower()):
#                 action_items.append({
#                     "task": task,
#                     "assigned_to": person,
#                     "due_date": "Not specified",
#                     "notes": f"Extracted from transcript"
#                 })
    
#     return action_items

# def extract_decisions(text):
#     """Extract likely decisions from the transcript"""
#     # Look for common decision patterns
#     decision_patterns = [
#         r'(?:we|they|it was)\s+decided\s+(?:to|that)\s+(.*?)(?:\.|,|$)',
#         r'(?:the)?\s*decision\s+(?:is|was)\s+(?:to|that)\s+(.*?)(?:\.|,|$)',
#         r'agreed\s+(?:to|that)\s+(.*?)(?:\.|,|$)',
#         r'(?:we|they)\s+will\s+(.*?)(?:\.|,|$)',
#         r'let\'s\s+(.*?)(?:\.|,|$)'
#     ]
    
#     decisions = []
    
#     for pattern in decision_patterns:
#         matches = re.findall(pattern, text, re.IGNORECASE)
#         for match in matches:
#             decision = match.strip()
#             if len(decision) > 10:  # Ensure meaningful content
#                 decisions.append(decision)
    
#     return decisions

# async def save_meeting_minutes(minutes_data, format="md"):
#     """
#     Save meeting minutes in markdown or text format.
    
#     Args:
#         minutes_data: Meeting minutes data structure
#         format: Output format ('md' for markdown, 'txt' for text)
        
#     Returns:
#         Path to the saved file
#     """
#     try:
#         # Create meetings directory if it doesn't exist
#         meetings_dir = os.path.join(OUTPUT_DIR, "meetings")
#         os.makedirs(meetings_dir, exist_ok=True)
        
#         # Sanitize title for filename
#         title = minutes_data.get('title', 'Meeting Minutes').replace("Meeting Minutes: ", "")
#         safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        
#         # Determine file extension
#         ext = ".md" if format.lower() == "md" else ".txt"
        
#         # Create the output file
#         output_path = os.path.join(meetings_dir, f"{safe_title}{ext}")
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             if format.lower() == "md":
#                 # Format as Markdown
#                 f.write(f"# {minutes_data.get('title', 'Meeting Minutes')}\n\n")
#                 f.write(f"**Date:** {minutes_data.get('date', 'Not specified')}\n")
#                 f.write(f"**Duration:** {minutes_data.get('duration', 'Not specified')}\n\n")
                
#                 # Participants
#                 f.write("## Participants\n\n")
#                 for participant in minutes_data.get('participants', ['Not specified']):
#                     f.write(f"- {participant}\n")
#                 f.write("\n")
                
#                 # Check if this is a review meeting format
#                 if 'agenda_items' in minutes_data:
#                     # Agenda Items
#                     f.write("## Agenda Items\n\n")
#                     for i, item in enumerate(minutes_data.get('agenda_items', [])):
#                         topic = item.get('topic', f'Item {i+1}')
#                         f.write(f"### {topic}\n\n")
#                         f.write(f"{item.get('discussion', 'No details provided')}\n\n")
#                         speakers = item.get('speakers', [])
#                         if speakers:
#                             f.write(f"**Speakers:** {', '.join(speakers)}\n\n")
                    
#                     # Key Metrics
#                     if 'key_metrics_discussed' in minutes_data and minutes_data['key_metrics_discussed']:
#                         f.write("## Key Metrics Discussed\n\n")
#                         for metric in minutes_data.get('key_metrics_discussed', []):
#                             f.write(f"### {metric.get('metric', 'Unnamed metric')}\n\n")
#                             f.write(f"**Status:** {metric.get('status', 'Not specified')}\n\n")
#                             f.write(f"{metric.get('details', 'No details provided')}\n\n")
#                 else:
#                     # Regular meeting format - Agenda
#                     f.write("## Agenda\n\n")
#                     for item in minutes_data.get('agenda', ['Not specified']):
#                         f.write(f"- {item}\n")
#                     f.write("\n")
                    
#                     # Key Points
#                     f.write("## Key Points\n\n")
#                     for point in minutes_data.get('key_points', []):
#                         topic = point.get('topic', 'Discussion')
#                         timestamp = point.get('timestamp', '')
#                         f.write(f"### {topic}")
#                         if timestamp:
#                             f.write(f" (at {timestamp})")
#                         f.write("\n\n")
                        
#                         for bullet in point.get('points', ['No specific points noted']):
#                             f.write(f"- {bullet}\n")
#                         f.write("\n")
                
#                 # Action Items
#                 f.write("## Action Items\n\n")
#                 if minutes_data.get('action_items'):
#                     for item in minutes_data.get('action_items', []):
#                         task = item.get('task', 'Undefined task')
#                         assignee = item.get('assigned_to', 'Unassigned')
#                         due_date = item.get('due_date', 'Not specified')
#                         notes = item.get('notes', '') or item.get('context', '')
                        
#                         f.write(f"- **Task:** {task}\n")
#                         f.write(f"  - **Assigned to:** {assignee}\n")
#                         f.write(f"  - **Due date:** {due_date}\n")
#                         if notes:
#                             f.write(f"  - **Notes/Context:** {notes}\n")
#                         f.write("\n")
#                 else:
#                     f.write("No specific action items noted.\n\n")
                
#                 # Decisions
#                 f.write("## Decisions\n\n")
#                 for decision in minutes_data.get('decisions', ['No specific decisions noted']):
#                     f.write(f"- {decision}\n")
#                 f.write("\n")
                
#                 # Next Steps or Next Meeting
#                 if 'next_steps' in minutes_data:
#                     f.write("## Next Steps\n\n")
#                     for step in minutes_data.get('next_steps', ['Not specified']):
#                         f.write(f"- {step}\n")
#                 else:
#                     f.write("## Next Meeting\n\n")
#                     f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
                
#             else:
#                 # Format as plain text - handling both meeting types
#                 f.write(f"{minutes_data.get('title', 'Meeting Minutes')}\n")
#                 f.write(f"Date: {minutes_data.get('date', 'Not specified')}\n")
#                 f.write(f"Duration: {minutes_data.get('duration', 'Not specified')}\n\n")
                
#                 # Participants
#                 f.write("PARTICIPANTS:\n")
#                 for participant in minutes_data.get('participants', ['Not specified']):
#                     f.write(f"- {participant}\n")
#                 f.write("\n")
                
#                 # Check if this is a review meeting
#                 if 'agenda_items' in minutes_data:
#                     # Agenda Items
#                     f.write("AGENDA ITEMS:\n")
#                     for i, item in enumerate(minutes_data.get('agenda_items', [])):
#                         topic = item.get('topic', f'Item {i+1}')
#                         f.write(f"{topic}:\n")
#                         f.write(f"{item.get('discussion', 'No details provided')}\n")
#                         speakers = item.get('speakers', [])
#                         if speakers:
#                             f.write(f"Speakers: {', '.join(speakers)}\n")
#                         f.write("\n")
                    
#                     # Key Metrics
#                     if 'key_metrics_discussed' in minutes_data and minutes_data['key_metrics_discussed']:
#                         f.write("KEY METRICS:\n")
#                         for metric in minutes_data.get('key_metrics_discussed', []):
#                             f.write(f"{metric.get('metric', 'Unnamed metric')}:\n")
#                             f.write(f"Status: {metric.get('status', 'Not specified')}\n")
#                             f.write(f"{metric.get('details', 'No details provided')}\n\n")
#                 else:
#                     # Regular meeting format - Agenda
#                     f.write("AGENDA:\n")
#                     for item in minutes_data.get('agenda', ['Not specified']):
#                         f.write(f"- {item}\n")
#                     f.write("\n")
                    
#                     # Key Points
#                     f.write("KEY POINTS:\n")
#                     for point in minutes_data.get('key_points', []):
#                         topic = point.get('topic', 'Discussion')
#                         timestamp = point.get('timestamp', '')
#                         if timestamp:
#                             f.write(f"{topic} (at {timestamp}):\n")
#                         else:
#                             f.write(f"{topic}:\n")
                        
#                         for bullet in point.get('points', ['No specific points noted']):
#                             f.write(f"- {bullet}\n")
#                         f.write("\n")
                
#                 # Action Items
#                 f.write("ACTION ITEMS:\n")
#                 if minutes_data.get('action_items'):
#                     for item in minutes_data.get('action_items', []):
#                         task = item.get('task', 'Undefined task')
#                         assignee = item.get('assigned_to', 'Unassigned')
#                         due_date = item.get('due_date', 'Not specified')
#                         notes = item.get('notes', '') or item.get('context', '')
                        
#                         f.write(f"- Task: {task}\n")
#                         f.write(f"  Assigned to: {assignee}\n")
#                         f.write(f"  Due date: {due_date}\n")
#                         if notes:
#                             f.write(f"  Notes/Context: {notes}\n")
#                         f.write("\n")
#                 else:
#                     f.write("No specific action items noted.\n\n")
                
#                 # Decisions
#                 f.write("DECISIONS:\n")
#                 for decision in minutes_data.get('decisions', ['No specific decisions noted']):
#                     f.write(f"- {decision}\n")
#                 f.write("\n")
                
#                 # Next Steps or Next Meeting
#                 if 'next_steps' in minutes_data:
#                     f.write("NEXT STEPS:\n")
#                     for step in minutes_data.get('next_steps', ['Not specified']):
#                         f.write(f"- {step}\n")
#                 else:
#                     f.write("NEXT MEETING:\n")
#                     f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
        
#         logger.info(f"Meeting minutes saved to: {output_path}")
#         return output_path
    
#     except Exception as e:
#         logger.error(f"Error saving meeting minutes: {e}")
#         return None

# def format_minutes_for_display(minutes_data):
#     """
#     Format meeting minutes for console display.
    
#     Args:
#         minutes_data: Meeting minutes data structure
        
#     Returns:
#         Formatted string for console output
#     """
#     output = []
    
#     # Title and basic info
#     output.append(f"\n{'=' * 80}")
#     output.append(f"  {minutes_data.get('title', 'MEETING MINUTES')}")
#     output.append(f"{'=' * 80}")
#     output.append(f"Date: {minutes_data.get('date', 'Not specified')}")
#     output.append(f"Duration: {minutes_data.get('duration', 'Not specified')}")
#     output.append("")
    
#     # Participants
#     output.append("PARTICIPANTS:")
#     for participant in minutes_data.get('participants', ['Not specified']):
#         output.append(f"• {participant}")
#     output.append("")
    
#     # Check if this is a review meeting with agenda_items
#     if 'agenda_items' in minutes_data:
#         # Agenda Items
#         output.append("AGENDA ITEMS:")
#         for item in minutes_data.get('agenda_items', []):
#             topic = item.get('topic', 'Discussion')
#             output.append(f"• {topic}:")
#             output.append(f"  - {item.get('discussion', 'No details provided')[:120]}...")
#             speakers = item.get('speakers', [])
#             if speakers:
#                 output.append(f"  - Speakers: {', '.join(speakers)}")
#         output.append("")
        
#         # Key Metrics
#         if 'key_metrics_discussed' in minutes_data and minutes_data['key_metrics_discussed']:
#             output.append("KEY METRICS:")
#             for metric in minutes_data.get('key_metrics_discussed', []):
#                 output.append(f"• {metric.get('metric', 'Unnamed metric')}: {metric.get('status', '')}")
#                 output.append(f"  - {metric.get('details', 'No details provided')[:100]}...")
#             output.append("")
#     else:
#         # Regular meeting format - Agenda
#         output.append("AGENDA:")
#         for item in minutes_data.get('agenda', ['Not specified']):
#             output.append(f"• {item}")
#         output.append("")
        
#         # Key Points (summarized)
#         output.append("KEY POINTS:")
#         for point in minutes_data.get('key_points', []):
#             topic = point.get('topic', 'Discussion')
#             timestamp = point.get('timestamp', '')
#             if timestamp:
#                 output.append(f"• {topic} (at {timestamp}):")
#             else:
#                 output.append(f"• {topic}:")
                
#             # Show just the first 2-3 points for each topic to keep display compact
#             points = point.get('points', ['No specific points noted'])
#             for i, bullet in enumerate(points):
#                 if i < 3:  # Show first 3 points
#                     output.append(f"  - {bullet}")
#                 elif i == 3:  # Indicate there are more points
#                     output.append(f"  - ... ({len(points) - 3} more points)")
#                     break
#         output.append("")
    
#     # Action Items
#     output.append("ACTION ITEMS:")
#     if minutes_data.get('action_items'):
#         for item in minutes_data.get('action_items', []):
#             task = item.get('task', 'Undefined task')
#             assignee = item.get('assigned_to', 'Unassigned')
#             due_date = item.get('due_date', 'Not specified')
#             notes = item.get('notes', '') or item.get('context', '')
            
#             output.append(f"• {task} → {assignee}")
#             if due_date and due_date != 'Not specified':
#                 output.append(f"  - Due: {due_date}")
#             if notes:
#                 output.append(f"  - Context: {notes[:80]}")
#     else:
#         output.append("• No specific action items noted.")
#     output.append("")
    
#     # Decisions (summarized)
#     output.append("DECISIONS:")
#     decisions = minutes_data.get('decisions', ['No specific decisions noted'])
#     for i, decision in enumerate(decisions):
#         if i < 3:  # Show first 3 decisions
#             output.append(f"• {decision}")
#         elif i == 3:  # Indicate there are more decisions
#             output.append(f"• ... ({len(decisions) - 3} more decisions)")
#             break
#     output.append("")
    
#     # Next Meeting or Next Steps
#     if 'next_steps' in minutes_data:
#         output.append("NEXT STEPS:")
#         for step in minutes_data.get('next_steps', ['Not specified']):
#             output.append(f"• {step}")
#     else:
#         output.append("NEXT MEETING:")
#         output.append(f"{minutes_data.get('next_meeting', 'Not specified')}")
#     output.append("")
    
#     # Join all lines
#     return "\n".join(output)

import re
import time
import os
import json
import ollama
from logger_config import logger
from constants import OUTPUT_DIR

async def generate_meeting_minutes(transcript_text, meeting_title="Meeting"):
    """
    Generate structured meeting minutes from a meeting transcript.
    
    Args:
        transcript_text: Full text of the meeting transcript
        meeting_title: Title of the meeting (default: "Meeting")
        
    Returns:
        Dictionary containing structured meeting minutes
    """
    start_time = time.time()
    
    # Create a prompt for the LLM
    prompt = f"""
    You are an expert meeting minutes generator. Given the transcript of a meeting, create professional, 
    structured meeting minutes that capture all essential information.
    
    MEETING DETAILS:
    - Title: "{meeting_title}"
    
    VERY IMPORTANT INSTRUCTIONS:
    1. Extract all participants from the transcript
    2. Identify the main topics discussed
    3. Capture key points for each topic
    4. Record all decisions made
    5. Identify all action items and who they are assigned to
    
    TRANSCRIPT CONTENT:
    {transcript_text}
    
    Generate comprehensive meeting minutes in this JSON structure:
    {{
      "title": "Meeting Minutes: {meeting_title}",
      "date": "Extract meeting date from transcript or use 'Not specified'",
      "participants": [
        "List all participants mentioned in the transcript"
      ],
      "agenda": [
        "First main topic discussed in the meeting",
        "Second main topic discussed in the meeting"
      ],
      "key_points": [
        {{
          "topic": "First important topic",
          "points": [
            "First key point about this topic",
            "Second key point about this topic"
          ]
        }},
        {{
          "topic": "Second important topic",
          "points": [
            "First key point about this topic",
            "Second key point about this topic"
          ]
        }}
      ],
      "action_items": [
        {{
          "task": "Specific task to be done",
          "assigned_to": "Person/team assigned",
          "due_date": "Due date if mentioned, otherwise 'Not specified'"
        }}
      ],
      "decisions": [
        "First decision made during the meeting",
        "Second decision made during the meeting"
      ],
      "next_meeting": "Details about the next meeting if mentioned, otherwise 'Not specified'"
    }}
    
    IMPORTANT GUIDELINES:
    1. Focus on accuracy - only include information from the transcript
    2. Format your response as VALID JSON only - this is critical
    3. Return only the JSON object, nothing else
    4. Be thorough and comprehensive - capture all important meeting content
    """
    
    try:
        logger.info("Generating meeting minutes with LLM...")
        
        # Call the LLM with our prompt
        response = ollama.chat(
            model="llama3:8b",  # You can use any model you prefer
            messages=[
                {"role": "system", "content": "You are a meeting summarization assistant. You create detailed, comprehensive meeting minutes in JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        end_time = time.time()
        logger.info(f"Time taken to generate meeting minutes: {end_time - start_time:.4f} seconds")
        
        # Extract the content from the response
        raw_content = response["message"]["content"]
        
        # Extract JSON from the response
        try:
            # Find JSON content between braces
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = raw_content[start_idx:end_idx+1]
                minutes_data = json.loads(json_str)
                return minutes_data
            else:
                logger.warning("Could not find JSON in the response")
                return {"error": "Failed to extract JSON from LLM response"}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return {"error": f"Failed to parse meeting minutes as JSON: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error generating meeting minutes: {e}")
        return {"error": f"Failed to generate meeting minutes: {str(e)}"}

# async def save_meeting_minutes(minutes_data, format="md"):
#     """
#     Save meeting minutes in markdown or text format.
    
#     Args:
#         minutes_data: Meeting minutes data structure
#         format: Output format ('md' for markdown, 'txt' for text)
        
#     Returns:
#         Path to the saved file
#     """
#     try:
#         # Create meetings directory if it doesn't exist
#         meetings_dir = os.path.join(OUTPUT_DIR, "meetings")
#         os.makedirs(meetings_dir, exist_ok=True)
        
#         # Get title for the filename
#         title = minutes_data.get('title', 'Meeting Minutes').replace("Meeting Minutes: ", "")
#         safe_title = title.strip().replace(' ', '_')
        
#         # Determine file extension
#         ext = ".md" if format.lower() == "md" else ".txt"
        
#         # Create the output file path
#         output_path = os.path.join(meetings_dir, f"{safe_title}{ext}")
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             if format.lower() == "md":
#                 # Format as Markdown
#                 f.write(f"# {minutes_data.get('title', 'Meeting Minutes')}\n\n")
#                 f.write(f"**Date:** {minutes_data.get('date', 'Not specified')}\n\n")
                
#                 # Participants
#                 f.write("## Participants\n\n")
#                 for participant in minutes_data.get('participants', ['Not specified']):
#                     f.write(f"- {participant}\n")
#                 f.write("\n")
                
#                 # Agenda
#                 f.write("## Agenda\n\n")
#                 for item in minutes_data.get('agenda', ['Not specified']):
#                     f.write(f"- {item}\n")
#                 f.write("\n")
                
#                 # Key Points
#                 f.write("## Key Points\n\n")
#                 for point in minutes_data.get('key_points', []):
#                     f.write(f"### {point.get('topic', 'Discussion')}\n\n")
#                     for bullet in point.get('points', ['No specific points noted']):
#                         f.write(f"- {bullet}\n")
#                     f.write("\n")
                
#                 # Action Items
#                 f.write("## Action Items\n\n")
#                 for item in minutes_data.get('action_items', []):
#                     task = item.get('task', 'Undefined task')
#                     assignee = item.get('assigned_to', 'Unassigned')
#                     due_date = item.get('due_date', 'Not specified')
                    
#                     f.write(f"- **Task:** {task}\n")
#                     f.write(f"  - **Assigned to:** {assignee}\n")
#                     f.write(f"  - **Due date:** {due_date}\n")
#                 f.write("\n")
                
#                 # Decisions
#                 f.write("## Decisions\n\n")
#                 for decision in minutes_data.get('decisions', ['No specific decisions noted']):
#                     f.write(f"- {decision}\n")
#                 f.write("\n")
                
#                 # Next Meeting
#                 f.write("## Next Meeting\n\n")
#                 f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
                
#             else:
#                 # Format as plain text
#                 f.write(f"{minutes_data.get('title', 'Meeting Minutes')}\n")
#                 f.write(f"Date: {minutes_data.get('date', 'Not specified')}\n\n")
                
#                 # Participants
#                 f.write("PARTICIPANTS:\n")
#                 for participant in minutes_data.get('participants', ['Not specified']):
#                     f.write(f"- {participant}\n")
#                 f.write("\n")
                
#                 # Agenda
#                 f.write("AGENDA:\n")
#                 for item in minutes_data.get('agenda', ['Not specified']):
#                     f.write(f"- {item}\n")
#                 f.write("\n")
                
#                 # Key Points
#                 f.write("KEY POINTS:\n")
#                 for point in minutes_data.get('key_points', []):
#                     f.write(f"{point.get('topic', 'Discussion')}:\n")
#                     for bullet in point.get('points', ['No specific points noted']):
#                         f.write(f"- {bullet}\n")
#                     f.write("\n")
                
#                 # Action Items
#                 f.write("ACTION ITEMS:\n")
#                 for item in minutes_data.get('action_items', []):
#                     task = item.get('task', 'Undefined task')
#                     assignee = item.get('assigned_to', 'Unassigned')
#                     due_date = item.get('due_date', 'Not specified')
                    
#                     f.write(f"- Task: {task}\n")
#                     f.write(f"  Assigned to: {assignee}\n")
#                     f.write(f"  Due date: {due_date}\n")
#                 f.write("\n")
                
#                 # Decisions
#                 f.write("DECISIONS:\n")
#                 for decision in minutes_data.get('decisions', ['No specific decisions noted']):
#                     f.write(f"- {decision}\n")
#                 f.write("\n")
                
#                 # Next Meeting
#                 f.write("NEXT MEETING:\n")
#                 f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
        
#         logger.info(f"Meeting minutes saved to: {output_path}")
#         return output_path
    
#     except Exception as e:
#         logger.error(f"Error saving meeting minutes: {e}")
#         return None


async def save_meeting_minutes(minutes_data, format="md"):
    """
    Save meeting minutes in markdown or text format.
    
    Args:
        minutes_data: Meeting minutes data structure
        format: Output format ('md' for markdown, 'txt' for text)
        
    Returns:
        Path to the saved file
    """
    try:
        # Create meetings directory if it doesn't exist
        meetings_dir = os.path.join(OUTPUT_DIR, "meetings")
        os.makedirs(meetings_dir, exist_ok=True)
        
        # Get title for the filename
        title = minutes_data.get('title', 'Meeting Minutes').replace("Meeting Minutes: ", "")
        
        # Properly sanitize the filename - replace any invalid characters with underscores
        # Windows specifically forbids: < > : " / \ | ? *
        safe_title = re.sub(r'[<>:"\\|?*\/]', '_', title.strip())
        safe_title = safe_title.replace(' ', '_')
        
        # Make sure filename is not too long
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        
        # Determine file extension
        ext = ".md" if format.lower() == "md" else ".txt"
        
        # Create the output file path
        output_path = os.path.join(meetings_dir, f"{safe_title}{ext}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() == "md":
                # Format as Markdown
                f.write(f"# {minutes_data.get('title', 'Meeting Minutes')}\n\n")
                f.write(f"**Date:** {minutes_data.get('date', 'Not specified')}\n\n")
                
                # Participants
                f.write("## Participants\n\n")
                for participant in minutes_data.get('participants', ['Not specified']):
                    f.write(f"- {participant}\n")
                f.write("\n")
                
                # Agenda
                f.write("## Agenda\n\n")
                for item in minutes_data.get('agenda', ['Not specified']):
                    f.write(f"- {item}\n")
                f.write("\n")
                
                # Key Points
                f.write("## Key Points\n\n")
                for point in minutes_data.get('key_points', []):
                    f.write(f"### {point.get('topic', 'Discussion')}\n\n")
                    for bullet in point.get('points', ['No specific points noted']):
                        f.write(f"- {bullet}\n")
                    f.write("\n")
                
                # Action Items
                f.write("## Action Items\n\n")
                for item in minutes_data.get('action_items', []):
                    task = item.get('task', 'Undefined task')
                    assignee = item.get('assigned_to', 'Unassigned')
                    due_date = item.get('due_date', 'Not specified')
                    
                    f.write(f"- **Task:** {task}\n")
                    f.write(f"  - **Assigned to:** {assignee}\n")
                    f.write(f"  - **Due date:** {due_date}\n")
                f.write("\n")
                
                # Decisions
                f.write("## Decisions\n\n")
                for decision in minutes_data.get('decisions', ['No specific decisions noted']):
                    f.write(f"- {decision}\n")
                f.write("\n")
                
                # Next Meeting
                f.write("## Next Meeting\n\n")
                f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
                
            else:
                # Format as plain text
                f.write(f"{minutes_data.get('title', 'Meeting Minutes')}\n")
                f.write(f"Date: {minutes_data.get('date', 'Not specified')}\n\n")
                
                # Participants
                f.write("PARTICIPANTS:\n")
                for participant in minutes_data.get('participants', ['Not specified']):
                    f.write(f"- {participant}\n")
                f.write("\n")
                
                # Agenda
                f.write("AGENDA:\n")
                for item in minutes_data.get('agenda', ['Not specified']):
                    f.write(f"- {item}\n")
                f.write("\n")
                
                # Key Points
                f.write("KEY POINTS:\n")
                for point in minutes_data.get('key_points', []):
                    f.write(f"{point.get('topic', 'Discussion')}:\n")
                    for bullet in point.get('points', ['No specific points noted']):
                        f.write(f"- {bullet}\n")
                    f.write("\n")
                
                # Action Items
                f.write("ACTION ITEMS:\n")
                for item in minutes_data.get('action_items', []):
                    task = item.get('task', 'Undefined task')
                    assignee = item.get('assigned_to', 'Unassigned')
                    due_date = item.get('due_date', 'Not specified')
                    
                    f.write(f"- Task: {task}\n")
                    f.write(f"  Assigned to: {assignee}\n")
                    f.write(f"  Due date: {due_date}\n")
                f.write("\n")
                
                # Decisions
                f.write("DECISIONS:\n")
                for decision in minutes_data.get('decisions', ['No specific decisions noted']):
                    f.write(f"- {decision}\n")
                f.write("\n")
                
                # Next Meeting
                f.write("NEXT MEETING:\n")
                f.write(f"{minutes_data.get('next_meeting', 'Not specified')}\n")
        
        logger.info(f"Meeting minutes saved to: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving meeting minutes: {e}")
        return None

# Example usage in main.py
async def main():
    # Load transcript from file
    with open("meeting_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()
    
    # Generate meeting minutes
    minutes = await generate_meeting_minutes(transcript, "Weekly Team Sync")
    
    # Save to file
    output_path = await save_meeting_minutes(minutes, format="md")
    
    # Display confirmation
    if output_path:
        print(f"Meeting minutes saved to: {output_path}")
    else:
        print("Failed to save meeting minutes.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())