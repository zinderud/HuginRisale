import json
from collections import OrderedDict

def process_youtube_links():
    # Read URLs from file
    with open('youtube.txt', 'r') as f:
        urls = f.readlines()
    
    # Track initial count
    initial_count = len(urls)
    
    # Remove duplicates while keeping order
    unique_urls = list(OrderedDict.fromkeys([url.strip() for url in urls if url.strip()]))
    
    # Track final count
    final_count = len(unique_urls)
    
    # Print duplicate stats
    duplicates = initial_count - final_count
    print(f"Initial URLs: {initial_count}")
    print(f"Unique URLs: {final_count}")
    print(f"Duplicates removed: {duplicates}")
    
    # Write unique URLs back to txt
    with open('youtube.txt', 'w') as f:
        f.write('\n'.join(unique_urls))
        
    # Write JSON format 
    with open('youtube.json', 'w') as f:
        json.dump(unique_urls, f, indent=2)

if __name__ == "__main__":
    process_youtube_links()
