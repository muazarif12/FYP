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
            model="deepseek-r1:7b",  # You can use any model you prefer
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