# COGREPO Explorer UI

An elegant web interface for querying and exploring your COGREPO conversation data.

## Features

### 🔍 **Advanced Search Capabilities**
- **Smart Search**: Search across all your conversations by keywords, phrases, or topics
- **Advanced Operators**: Use `"exact phrases"`, `+required` terms, `-excluded` words
- **Full-text Search with Ranking**: Results scored by relevance with TF-IDF weighting
- **🧠 Semantic Search**: Find conversations by meaning, not just keywords (e.g., search "money" finds "budget", "finance", "investment")
- **⚡ Real-time Search**: Search as you type with smart debouncing (500ms delay)

### 📌 **Saved Searches & Bookmarks**
- **Save Frequent Queries**: Bookmark your most-used search terms
- **Quick Access**: One-click access to saved searches
- **Local Storage**: Searches persist between sessions

### 🎛️ **Enhanced Filtering & UI**
- **Quick Filters**: Pre-configured buttons for common searches (Family Travel, Business, etc.)
- **Date Range Filtering**: Find conversations from specific time periods  
- **Source Filtering**: Filter by AI provider (OpenAI, Anthropic, Google)
- **📊 Statistics Panel**: See insights about your search results
- **💾 Export Results**: Download search results as JSON
- **🎨 Beautiful UI**: Modern, gradient-based design with smooth animations
- **📱 Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Option 1: Simple HTML Version (No Server)
1. Open `index.html` in your browser
2. The interface will load with sample data
3. Start searching!

### Option 2: With Python Backend (Real Data)
1. Start the backend server:
   ```bash
   python3 server.py
   ```
2. Open `index.html` in your browser
3. The UI will now load your actual COGREPO data

### Option 3: Build with Claude Code
1. In your terminal, navigate to this directory:
   ```bash
   cd ~/Desktop/cogrepo-ui
   ```
2. Start Claude Code:
   ```bash
   claude
   ```
3. Ask Claude to help you enhance the UI or add features

## Customization

### Adding Quick Filters
Edit the quick filter buttons in `index.html`:
```html
<button class="quick-filter" onclick="quickSearch('your search term')">Your Label</button>
```

### Changing Data Source
Edit the `REPO_PATH` in `server.py`:
```python
REPO_PATH = "/path/to/your/enriched_repository.jsonl"
```

### Styling
All styles are in the `<style>` section of `index.html`. Key colors:
- Primary gradient: `#667eea` to `#764ba2`
- Customize by changing these hex values

## File Structure

```
cogrepo-ui/
├── index.html       # Main UI interface
├── server.py        # Python backend API server
├── package.json     # Node.js configuration (for Claude Code)
└── README.md        # This file
```

## Search Examples

### Basic Search
```
family vacation
```

### Advanced Operators
```
"exact phrase" +required -excluded
```
- `"family vacation"` - Find exact phrase
- `+travel +kids -business` - Must have travel AND kids, exclude business
- `"road trip" +planning -expensive` - Exact phrase with required/excluded terms

### Semantic Search Examples
Try the 🧠 **Semantic** button for these queries:
- `money` → finds: finance, budget, investment, salary, cost, payment
- `family` → finds: parent, child, kids, spouse, relatives
- `work` → finds: job, career, office, business, professional
- `health` → finds: medical, doctor, wellness, fitness, exercise
- `creative` → finds: art, design, music, writing, photography

### Saved Searches
1. Enter your search query
2. Click the 💾 **Save** button  
3. Give it a name like "Family Travel Plans"
4. Access it anytime from the 📌 **Saved Searches** panel

## Performance & Scoring

### Relevance Scoring
- **Title matches**: 3x weight boost
- **Summary matches**: 2x weight boost  
- **Required terms**: 2x weight boost
- **Exact phrases**: High fixed score (5 points)
- **TF-IDF weighting**: More mentions = higher score

### Semantic Scoring  
- **Original terms**: 3x weight (high priority)
- **Semantic expansions**: 1x weight (related concepts)
- **Multi-match bonus**: 20% boost for multiple concept matches
- **Field weighting**: Title > Summary > Full text

## Using with Claude Code

Claude Code can help you:
- Add React components for more interactivity
- Implement a proper backend with Node.js/Express
- Add a database for faster queries (SQLite, PostgreSQL)
- Create data visualizations with D3.js or Chart.js
- Build a full-stack application with authentication

Example Claude Code commands:
```bash
# Initialize a React app
claude "create a React version of this COGREPO explorer"

# Add search functionality
claude "add fuzzy search to find conversations even with typos"

# Create visualizations
claude "add a timeline visualization of my conversations"
```

## Troubleshooting

### Data not loading?
- Check that `REPO_PATH` in `server.py` points to your actual data file
- Ensure the Python server is running on port 8000
- Check browser console for errors (F12)

### Search not working?
- The simple HTML version uses sample data
- Run the Python server for real data
- Check that your search terms exist in the conversations

### UI looks broken?
- Use a modern browser (Chrome, Firefox, Safari, Edge)
- Clear browser cache and reload
- Check that all files are in the same directory

## Next Steps

1. **Test the current UI** with your data
2. **Identify missing features** you need
3. **Use Claude Code** to implement enhancements
4. **Consider a database** for large datasets (>10,000 conversations)
5. **Add user authentication** if sharing with others

## Support

For help building additional features, you can:
- Use Claude Code: `claude "help me add [feature] to COGREPO explorer"`
- Ask Claude in the chat for implementation guidance
- Check the browser console for debugging information

---

Built with ❤️ for better conversation discovery