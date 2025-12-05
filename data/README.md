# CogRepo Data Directory

This directory contains your processed conversation data from the CogRepo pipeline.

## 📊 Data Files Overview

### **Primary Database**
- **`enriched_repository.jsonl`** (165MB)
  - Complete processed conversation database
  - Each line = one conversation with AI analysis
  - Used by all search tools and web UI
  - Contains: raw text, summaries, tags, timestamps, sources

### **Analysis Outputs**  
- **`focus_list.jsonl`** (89MB)
  - High-priority conversations and action items
  - Filtered from main database based on strategic importance
  - Pre-processed for immediate attention

- **`strategic_projects.json`** (5KB)
  - Synthesized strategic insights and themes
  - Key projects and opportunities identified
  - Generated from pattern analysis across all conversations

### **Technical Files**
- **`repository.index`** (45MB)
  - Search embeddings for semantic search
  - Used by search tools for finding related concepts
  - Generated using AI embeddings

- **`repository.index.meta.json`**
  - Metadata about the search index
  - Index configuration and statistics

- **`standardized_conversations.parquet`** (80MB)
  - Structured data format optimized for analysis
  - Use with pandas, data science tools
  - Same data as JSONL but in columnar format

## 🔍 How to Use Each File

### **For Search & Exploration:**
Use `enriched_repository.jsonl` with:
- `../cogrepo_search.py "query"`
- `../cogrepo_date_search.py --start 2024-01-01`
- Web UI at `../cogrepo-ui/`

### **For Immediate Action:**
```bash
# View high-priority items
head -20 focus_list.jsonl | jq '.generated_title'

# Find specific priorities
grep -i "modern magic" focus_list.jsonl
```

### **For Strategic Planning:**
```bash
# View key insights
cat strategic_projects.json | jq '.'

# Extract themes
jq '.strategic_themes[].title' strategic_projects.json
```

### **For Data Analysis:**
```python
import pandas as pd
import json

# Load structured data
df = pd.read_parquet('standardized_conversations.parquet')

# Or load JSONL
conversations = []
with open('enriched_repository.jsonl', 'r') as f:
    for line in f:
        conversations.append(json.loads(line))
```

## 📈 File Sizes & Processing Stats

| File | Size | Records | Purpose |
|------|------|---------|---------|
| enriched_repository.jsonl | 165MB | 3,748 | Main database |
| focus_list.jsonl | 89MB | ~1,200 | Priority items |  
| strategic_projects.json | 5KB | 5 themes | Key insights |
| repository.index | 45MB | 3,748 | Search embeddings |
| standardized_conversations.parquet | 80MB | 3,748 | Analysis format |

**Total:** ~384MB processed from years of AI conversations

## 🔄 Data Updates

When you have new conversations to process:
1. Export new conversations from AI platforms
2. Run the processing pipeline
3. Files in this directory will be updated automatically
4. Search tools will use the refreshed data

## 🛡️ Data Security

⚠️ **Important Notes:**
- These files contain your personal conversation history
- Never commit to public repositories (.gitignore handles this)
- Back up separately from code repositories
- Consider encryption for sensitive conversations

---
*Data processed from 3,748 conversations spanning multiple AI platforms*