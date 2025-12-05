#!/usr/bin/env python3
"""
View COGREPO conversations by date range
"""

import json
import sys
from datetime import datetime, timedelta
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

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        # Parse the timestamp (format: "2022-12-12 02:17:40.899082899")
        return datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
    except:
        return None

def format_date(timestamp_str):
    """Format timestamp for display"""
    dt = parse_timestamp(timestamp_str)
    return dt.strftime("%Y-%m-%d") if dt else timestamp_str

def main():
    print("📅 COGREPO Date Range Search")
    print("-" * 40)
    
    # Get date range
    print("\nEnter date range (YYYY-MM-DD format)")
    print("Press Enter for defaults:")
    print("  - Start: Beginning of your conversations")
    print("  - End: Today")
    
    start_input = input("\nStart date (or press Enter): ").strip()
    end_input = input("End date (or press Enter): ").strip()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if start_input:
        try:
            start_date = datetime.strptime(start_input, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid start date format. Using no start limit.")
    
    if end_input:
        try:
            end_date = datetime.strptime(end_input, "%Y-%m-%d") + timedelta(days=1)  # Include the entire end day
        except ValueError:
            print(f"Invalid end date format. Using no end limit.")
    
    # Optional keyword filter
    keyword = input("\nOptional: Enter keyword to filter (or press Enter to skip): ").strip()
    
    # Check if file exists
    if not Path(REPO_PATH).exists():
        print(f"\n❌ Error: Could not find {REPO_PATH}")
        sys.exit(1)
    
    # Read and filter conversations
    matching_convos = []
    total_convos = 0
    date_stats = {}
    
    print("\n🔍 Searching conversations...")
    with open(REPO_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total_convos += 1
            try:
                convo = json.loads(line)
                convo_dt = parse_timestamp(convo.get('timestamp', ''))
                
                if not convo_dt:
                    continue
                
                # Check date range
                if start_date and convo_dt < start_date:
                    continue
                if end_date and convo_dt >= end_date:
                    continue
                
                # Check keyword if provided
                if keyword:
                    searchable = []
                    if 'raw_text' in convo:
                        searchable.append(convo['raw_text'].lower())
                    if 'generated_title' in convo:
                        searchable.append(convo['generated_title'].lower())
                    if 'summary_abstractive' in convo:
                        searchable.append(convo['summary_abstractive'].lower())
                    
                    full_text = ' '.join(searchable)
                    if keyword.lower() not in full_text:
                        continue
                
                matching_convos.append(convo)
                
                # Track stats by month
                month_key = convo_dt.strftime("%Y-%m")
                date_stats[month_key] = date_stats.get(month_key, 0) + 1
                
            except (json.JSONDecodeError, KeyError):
                continue
            
            # Progress indicator
            if total_convos % 500 == 0:
                print(f"  Processed {total_convos} conversations...")
    
    # Sort by date
    matching_convos.sort(key=lambda x: x.get('timestamp', ''))
    
    # Display summary statistics
    print(f"\n📊 Summary Statistics")
    print("=" * 80)
    print(f"Total conversations searched: {total_convos}")
    print(f"Matching conversations: {len(matching_convos)}")
    
    if date_stats:
        print(f"\n📅 Conversations by Month:")
        for month in sorted(date_stats.keys()):
            print(f"  {month}: {date_stats[month]} conversations")
    
    # Display results
    print(f"\n📝 Conversation List (sorted by date)")
    print("=" * 80)
    
    # Group by date for better readability
    current_date = None
    for convo in matching_convos:
        date = format_date(convo.get('timestamp', 'Unknown date'))
        title = convo.get('generated_title', 'Untitled')
        source = convo.get('source', 'Unknown')
        
        # Add date header when date changes
        if date != current_date:
            print(f"\n📅 {date}")
            print("-" * 40)
            current_date = date
        
        print(f"  • {title}")
        print(f"    Source: {source}")
        
        # Show tags if they exist
        tags = convo.get('tags', [])
        if tags:
            print(f"    Tags: {', '.join(tags[:5])}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"cogrepo_date_range_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matching_convos, f, indent=2)
    print(f"\n📁 Full results saved to: {output_path}")
    print(f"   ({len(matching_convos)} conversations)")

if __name__ == "__main__":
    main()
