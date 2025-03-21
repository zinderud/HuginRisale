from youtube_transcript_api import YouTubeTranscriptApi
import os

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if "v=" in url:
        return url.split("v=")[1][:11]
    return url.split("/")[-1][:11]

def get_turkish_subtitle(video_id):
    """Get Turkish subtitle for a video"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manual Turkish subtitle first
        try:
            transcript = transcript_list.find_transcript(['tr'])
        except:
            # If manual Turkish not available, try auto-translated
            try:
                transcript = transcript_list.find_manually_created_transcript()
                transcript = transcript.translate('tr')
            except:
                return None
        
        return transcript.fetch()
        
    except Exception as e:
        print(f"Error getting subtitles for {video_id}: {str(e)}")
        return None

def save_subtitle(subtitle_data, video_id):
    """Save subtitle to file"""
    if not subtitle_data:
        return
    
    # Create subtitles directory if not exists
    os.makedirs("subtitles", exist_ok=True)
    
    # Save as text file
    with open(f"subtitles/{video_id}.txt", "w", encoding="utf-8") as f:
        for entry in subtitle_data:
            # Use proper attribute access instead of dictionary access
            f.write(f"{entry.text}\n")

def main():
    # Read YouTube URLs
    with open("youtube.txt", "r") as f:
        urls = f.readlines()
    
    # Process each URL
    for url in urls:
        url = url.strip()
        if url:
            video_id = extract_video_id(url)
            print(f"Processing video: {video_id}")
            
            subtitle_data = get_turkish_subtitle(video_id)
            if subtitle_data:
                save_subtitle(subtitle_data, video_id)
                print(f"Saved subtitles for {video_id}")
            else:
                print(f"No Turkish subtitles found for {video_id}")

if __name__ == "__main__":
    main()
