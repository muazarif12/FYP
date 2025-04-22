import datetime

# Format seconds to HH:MM:SS format
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# def format_timestamp(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     seconds = int(seconds % 60)
#     if hours > 0:
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
#     else:
#         return f"{minutes:02d}:{seconds:02d}"


def format_time_duration(seconds):
    """Convert a number of seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=round(seconds)))