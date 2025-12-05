#!/usr/bin/env python3
"""
COGREPO SEARCH TOOL - Mine Your AI Conversation History

Purpose: Search through thousands of enriched LLM conversations to find specific topics,
         insights, or discussions from your AI interaction history.

How it works:
1. Loads enriched conversation data (with summaries, tags, and metadata)
2. Searches across multiple fields: raw text, titles, summaries, and tags
3. Returns matching conversations sorted chronologically
4. Saves results to JSON for further analysis

Usage:
    python cogrepo_search.py "search terms"   # Command line
    python cogrepo_search.py                   # Interactive mode

Real value: Rediscover forgotten insights, track idea evolution, find that brilliant
           solution you discussed months ago but can't quite remember.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Path to your enriched repository
def get_repo_path():
    """Get the repository path - checks multiple locations"""
    import os
    
    if 'COGREPO_PATH' in os.environ:
        return os.environ['COGREPO_PATH']
    
    possible_paths = [
        # If running from cogrepo directory
        "data/enriched_repository.jsonl",
        # If running from RR Coding Experiments root
        "cogrepo/data/enriched_repository.jsonl",
        # Fallback paths (legacy)
        "/Users/newuser/Desktop/COGREPO_FINAL_20250816/enriched_repository.jsonl",
        "/Users/newuser/Desktop/RR Coding Experiments/cogrepo/data/enriched_repository.jsonl"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return possible_paths[0]

REPO_PATH = get_repo_path()

def search_conversation(convo_data, search_query):
    """Check if conversation contains search query"""
    # Convert search fields to lowercase for case-insensitive search
    searchable_text = []
    
    # Add raw conversation text
    if 'raw_text' in convo_data:
        searchable_text.append(convo_data['raw_text'].lower())
    
    # Add title
    if 'generated_title' in convo_data:
        searchable_text.append(convo_data['generated_title'].lower())
    
    # Add summary
    if 'summary_abstractive' in convo_data:
        searchable_text.append(convo_data['summary_abstractive'].lower())
    
    # Add tags
    if 'tags' in convo_data and isinstance(convo_data['tags'], list):
        searchable_text.append(' '.join(convo_data['tags']).lower())
    
    # Combine all searchable text
    full_text = ' '.join(searchable_text)
    
    # Check if search query appears in the text
    return search_query.lower() in full_text

def format_date(timestamp_str):
    """Format timestamp for display"""
    try:
        # Parse the timestamp (format: "2022-12-12 02:17:40.899082899")
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")
    except:
        return timestamp_str

def main():
    # Get search query from command line or ask user
    if len(sys.argv) > 1:
        search_query = ' '.join(sys.argv[1:])
    else:
        print("🔍 COGREPO Conversation Search Tool")
        print("-" * 40)
        search_query = input("Enter search terms (e.g., 'family travel'): ").strip()
        if not search_query:
            print("No search terms provided. Exiting.")
            sys.exit(0)
    
    print(f"\n🔍 Searching for: '{search_query}'\n")
    
    # Check if file exists
    if not Path(REPO_PATH).exists():
        print(f"❌ Error: Could not find {REPO_PATH}")
        sys.exit(1)
    
    # Read and search conversations
    matching_convos = []
    total_convos = 0
    
    print("Searching conversations...")
    with open(REPO_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total_convos += 1
            try:
                convo = json.loads(line)
                if search_conversation(convo, search_query):
                    matching_convos.append(convo)
            except json.JSONDecodeError:
                continue
            
            # Progress indicator
            if total_convos % 500 == 0:
                print(f"  Processed {total_convos} conversations...")
    
    # Sort by date (timestamp field)
    matching_convos.sort(key=lambda x: x.get('timestamp', ''))
    
    # Display results
    print(f"\n✅ Found {len(matching_convos)} matching conversations out of {total_convos} total\n")
    print("=" * 80)
    
    # Ask how to display results
    if len(matching_convos) > 20:
        print(f"Found {len(matching_convos)} results. How would you like to view them?")
        print("1. Show all (full list)")
        print("2. Show first 20")
        print("3. Show last 20")
        print("4. Save to file only")
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '2':
            display_convos = matching_convos[:20]
        elif choice == '3':
            display_convos = matching_convos[-20:]
        elif choice == '4':
            display_convos = []
        else:
            display_convos = matching_convos
    else:
        display_convos = matching_convos
    
    # Display selected conversations
    for i, convo in enumerate(display_convos, 1):
        date = format_date(convo.get('timestamp', 'Unknown date'))
        title = convo.get('generated_title', 'Untitled')
        source = convo.get('source', 'Unknown')
        
        print(f"\n{i}. [{date}] {title}")
        print(f"   Source: {source}")
        
        # Show a brief excerpt from the summary
        summary = convo.get('summary_abstractive', '')
        if summary:
            # Show first 200 characters of summary
            excerpt = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"   {excerpt}")
        
        print("-" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"cogrepo_search_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matching_convos, f, indent=2)
    print(f"\n📁 Full results saved to: {output_path}")
    print(f"   ({len(matching_convos)} conversations)")

if __name__ == "__main__":
    main()
