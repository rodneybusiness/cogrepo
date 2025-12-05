#!/usr/bin/env python3
"""
Backend server for COGREPO Explorer UI
Serves conversation data via a simple HTTP API
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from urllib.parse import urlparse, parse_qs
import re
import math
from collections import defaultdict, Counter
from datetime import datetime

# Path to your COGREPO data
def get_repo_path():
    """Get the repository path - checks multiple locations"""
    import os
    from pathlib import Path
    
    if 'COGREPO_PATH' in os.environ:
        return os.environ['COGREPO_PATH']
    
    possible_paths = [
        # If running from cogrepo-ui directory
        "../data/enriched_repository.jsonl",
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

class COGREPOHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Enable CORS
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if parsed_path.path == '/api/conversations':
            # Load and return all conversations
            conversations = self.load_conversations()
            self.wfile.write(json.dumps(conversations).encode())
            
        elif parsed_path.path == '/api/search':
            # Parse query parameters
            params = parse_qs(parsed_path.query)
            query = params.get('q', [''])[0]
            source = params.get('source', [''])[0]
            date_from = params.get('from', [''])[0]
            date_to = params.get('to', [''])[0]
            
            # Search conversations
            results = self.search_conversations(query, source, date_from, date_to)
            self.wfile.write(json.dumps(results).encode())
            
        elif parsed_path.path.startswith('/api/conversation/'):
            # Get full conversation details by ID
            convo_id = parsed_path.path.split('/api/conversation/')[1]
            conversation = self.get_conversation_by_id(convo_id)
            self.wfile.write(json.dumps(conversation).encode())
            
        elif parsed_path.path == '/api/semantic-search':
            # Semantic search endpoint
            params = parse_qs(parsed_path.query)
            query = params.get('q', [''])[0]
            source = params.get('source', [''])[0]
            date_from = params.get('from', [''])[0]
            date_to = params.get('to', [''])[0]
            
            # Perform semantic search
            results = self.semantic_search_conversations(query, source, date_from, date_to)
            self.wfile.write(json.dumps(results).encode())
    
    def load_conversations(self):
        """Load all conversations from JSONL file"""
        conversations = []
        
        try:
            with open(REPO_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        convo = json.loads(line)
                        # Format for UI
                        formatted = {
                            'id': convo.get('convo_id', ''),
                            'date': convo.get('timestamp', '')[:10] if convo.get('timestamp') else '',
                            'title': convo.get('generated_title', 'Untitled'),
                            'summary': convo.get('summary_abstractive', '')[:300],
                            'source': convo.get('source', 'Unknown'),
                            'tags': convo.get('tags', [])[:5],
                            'text': ' '.join([
                                convo.get('raw_text', '')[:500],
                                convo.get('generated_title', ''),
                                convo.get('summary_abstractive', '')
                            ]).lower()
                        }
                        conversations.append(formatted)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: Could not find {REPO_PATH}")
        
        return conversations
    
    def search_conversations(self, query, source, date_from, date_to):
        """Advanced search with operators and ranking"""
        conversations = self.load_conversations()
        
        if not query.strip():
            # No query - apply filters only
            results = self._apply_filters(conversations, source, date_from, date_to)
            results.sort(key=lambda x: x['date'], reverse=True)
            return results
        
        # Parse advanced search operators
        search_terms = self._parse_search_query(query)
        
        # Score and filter conversations
        scored_results = []
        for convo in conversations:
            # Apply basic filters first
            if source and convo['source'] != source:
                continue
            if date_from and convo['date'] < date_from:
                continue
            if date_to and convo['date'] > date_to:
                continue
            
            # Calculate relevance score
            score = self._calculate_relevance_score(convo, search_terms)
            if score > 0:
                convo['relevance_score'] = score
                scored_results.append(convo)
        
        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_results
    
    def _parse_search_query(self, query):
        """Parse query into required, excluded, exact phrases, and regular terms"""
        terms = {
            'required': [],     # +term
            'excluded': [],     # -term  
            'phrases': [],      # "exact phrase"
            'regular': []       # regular terms
        }
        
        # Extract exact phrases first
        phrase_pattern = r'"([^"]*)"'
        phrases = re.findall(phrase_pattern, query)
        terms['phrases'] = [p.lower() for p in phrases if p.strip()]
        
        # Remove phrases from query for further processing
        query_no_phrases = re.sub(phrase_pattern, '', query)
        
        # Split remaining terms and categorize
        words = query_no_phrases.split()
        for word in words:
            word = word.strip()
            if not word:
                continue
                
            if word.startswith('+'):
                terms['required'].append(word[1:].lower())
            elif word.startswith('-'):
                terms['excluded'].append(word[1:].lower())
            else:
                terms['regular'].append(word.lower())
        
        return terms
    
    def _calculate_relevance_score(self, convo, search_terms):
        """Calculate TF-IDF style relevance score"""
        text_fields = [
            convo.get('title', ''),
            convo.get('summary', ''),
            convo.get('text', '')
        ]
        full_text = ' '.join(text_fields).lower()
        
        # Check excluded terms first - if any found, score is 0
        for excluded in search_terms['excluded']:
            if excluded in full_text:
                return 0
        
        # Check required terms - all must be present
        for required in search_terms['required']:
            if required not in full_text:
                return 0
        
        # Check exact phrases - all must be present
        for phrase in search_terms['phrases']:
            if phrase not in full_text:
                return 0
        
        score = 0
        
        # Score regular terms with TF-IDF weighting
        for term in search_terms['regular']:
            if term in full_text:
                # Term frequency in document
                tf = full_text.count(term)
                
                # Boost score for title matches
                title_boost = 3 if term in convo.get('title', '').lower() else 1
                
                # Boost score for summary matches  
                summary_boost = 2 if term in convo.get('summary', '').lower() else 1
                
                score += tf * title_boost * summary_boost
        
        # Score required terms (they must all be present to reach here)
        for term in search_terms['required']:
            tf = full_text.count(term)
            score += tf * 2  # Required terms get double weight
        
        # Score exact phrases (they must all be present to reach here)
        for phrase in search_terms['phrases']:
            score += 5  # Exact phrases get high fixed score
        
        return score
    
    def _apply_filters(self, conversations, source, date_from, date_to):
        """Apply source and date filters"""
        results = []
        for convo in conversations:
            if source and convo['source'] != source:
                continue
            if date_from and convo['date'] < date_from:
                continue
            if date_to and convo['date'] > date_to:
                continue
            results.append(convo)
        return results
    
    def semantic_search_conversations(self, query, source, date_from, date_to):
        """Semantic search using word similarity and synonyms"""
        conversations = self.load_conversations()
        
        if not query.strip():
            results = self._apply_filters(conversations, source, date_from, date_to)
            results.sort(key=lambda x: x['date'], reverse=True)
            return results
        
        # Simple semantic mapping for common concepts
        semantic_mappings = {
            'money': ['finance', 'budget', 'cost', 'price', 'payment', 'salary', 'investment', 'banking', 'financial'],
            'family': ['parent', 'child', 'spouse', 'relative', 'brother', 'sister', 'mother', 'father', 'kids'],
            'work': ['job', 'career', 'employment', 'office', 'business', 'professional', 'colleague', 'project'],
            'travel': ['trip', 'vacation', 'journey', 'flight', 'hotel', 'tourism', 'destination', 'visit'],
            'food': ['eat', 'restaurant', 'cooking', 'recipe', 'meal', 'dinner', 'lunch', 'breakfast'],
            'technology': ['computer', 'software', 'programming', 'coding', 'tech', 'digital', 'app', 'website'],
            'health': ['medical', 'doctor', 'hospital', 'medicine', 'wellness', 'fitness', 'exercise', 'therapy'],
            'education': ['school', 'learning', 'study', 'course', 'university', 'teaching', 'academic', 'student'],
            'creative': ['art', 'design', 'music', 'writing', 'painting', 'photography', 'creative', 'artistic'],
            'personal': ['private', 'individual', 'myself', 'personal', 'self', 'own', 'me', 'my']
        }
        
        # Expand query with semantic terms
        query_lower = query.lower()
        expanded_terms = [query_lower]
        
        for concept, synonyms in semantic_mappings.items():
            if concept in query_lower or any(syn in query_lower for syn in synonyms):
                expanded_terms.extend(synonyms)
        
        # Score conversations based on semantic similarity
        scored_results = []
        for convo in conversations:
            # Apply basic filters first
            if source and convo['source'] != source:
                continue
            if date_from and convo['date'] < date_from:
                continue
            if date_to and convo['date'] > date_to:
                continue
            
            # Calculate semantic similarity score
            score = self._calculate_semantic_score(convo, expanded_terms, query_lower)
            if score > 0:
                convo['semantic_score'] = score
                scored_results.append(convo)
        
        # Sort by semantic score (highest first)
        scored_results.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        return scored_results[:50]  # Limit to top 50 results
    
    def _calculate_semantic_score(self, convo, expanded_terms, original_query):
        """Calculate semantic similarity score"""
        text_fields = [
            convo.get('title', ''),
            convo.get('summary', ''),
            convo.get('text', '')
        ]
        full_text = ' '.join(text_fields).lower()
        
        score = 0
        
        # Higher weight for original query terms
        original_words = original_query.split()
        for word in original_words:
            if word in full_text:
                count = full_text.count(word)
                # Boost for title/summary matches
                title_boost = 5 if word in convo.get('title', '').lower() else 1
                summary_boost = 3 if word in convo.get('summary', '').lower() else 1
                score += count * title_boost * summary_boost * 3  # Triple weight for original terms
        
        # Lower weight for semantic expansion terms
        for term in expanded_terms[1:]:  # Skip first term (original query)
            if term in full_text:
                count = full_text.count(term)
                title_boost = 2 if term in convo.get('title', '').lower() else 1
                summary_boost = 1.5 if term in convo.get('summary', '').lower() else 1
                score += count * title_boost * summary_boost  # Normal weight for semantic terms
        
        # Bonus for multiple semantic matches
        matches = sum(1 for term in expanded_terms if term in full_text)
        if matches > 2:
            score *= 1.2  # 20% bonus for multiple matches
            
        return score
    
    def get_conversation_by_id(self, convo_id):
        """Get full conversation details by ID"""
        try:
            with open(REPO_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        convo = json.loads(line)
                        if convo.get('convo_id') == convo_id:
                            # Return full conversation with complete text
                            return {
                                'id': convo.get('convo_id', ''),
                                'date': convo.get('timestamp', '')[:10] if convo.get('timestamp') else '',
                                'title': convo.get('generated_title', 'Untitled'),
                                'summary': convo.get('summary_abstractive', ''),
                                'source': convo.get('source', 'Unknown'),
                                'tags': convo.get('tags', []),
                                'full_text': convo.get('raw_text', ''),
                                'timestamp': convo.get('timestamp', ''),
                                'summary_extractive': convo.get('summary_extractive', ''),
                                'key_topics': convo.get('key_topics', [])
                            }
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: Could not find {REPO_PATH}")
        
        return {'error': 'Conversation not found'}
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, COGREPOHandler)
    print(f"🚀 COGREPO API Server running on http://localhost:{port}")
    print(f"   Open index.html in your browser to use the UI")
    print(f"   Press Ctrl+C to stop")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()