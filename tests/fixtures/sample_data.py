#!/usr/bin/env python3
"""
Sample Data Generator for CogRepo

Creates realistic sample conversations for testing and development.
Covers multiple sources, domains, and content types.

Usage:
    python -m tests.fixtures.sample_data --output data/enriched_repository.jsonl
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# Sample Conversation Templates
# =============================================================================

CONVERSATIONS = [
    # Technical - Python/Async
    {
        "domain": "Technical",
        "tags": ["python", "async", "aiohttp", "concurrency"],
        "title": "Implementing Async HTTP Client in Python",
        "messages": [
            ("user", "How do I make concurrent HTTP requests in Python? I need to fetch data from multiple APIs."),
            ("assistant", """For concurrent HTTP requests in Python, I recommend using `aiohttp` with `asyncio`. Here's a complete example:

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = ['https://api.example.com/1', 'https://api.example.com/2']
results = asyncio.run(fetch_all(urls))
```

Key points:
1. `aiohttp.ClientSession` manages connection pooling
2. `asyncio.gather` runs coroutines concurrently
3. Context managers ensure proper cleanup"""),
            ("user", "What if one of the requests fails? I don't want the whole batch to fail."),
            ("assistant", """Great question! Use `return_exceptions=True` in `asyncio.gather`:

```python
async def fetch_all_safe(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                print(f"Failed: {url} - {result}")
            else:
                successful.append(result)
        return successful
```

This way, exceptions are returned as values instead of raised, and your other requests complete normally.""")
        ],
        "score": 8,
        "insights": ["aiohttp is preferred for async HTTP", "asyncio.gather runs concurrent coroutines", "return_exceptions=True for fault tolerance"]
    },

    # Technical - Docker
    {
        "domain": "Technical",
        "tags": ["docker", "containers", "devops", "redis"],
        "title": "Setting Up Redis with Docker for Local Development",
        "messages": [
            ("user", "I need to run Redis locally for development. What's the best way with Docker?"),
            ("assistant", """Here's the simplest way to run Redis with Docker:

```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

For persistent data, add a volume:

```bash
docker run -d --name redis \\
  -p 6379:6379 \\
  -v redis_data:/data \\
  redis:alpine --appendonly yes
```

The `--appendonly yes` flag enables persistence so data survives container restarts.

To connect from your app:
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.set('key', 'value')
```"""),
            ("user", "How do I check if it's running and see the logs?"),
            ("assistant", """Several useful commands:

```bash
# Check if running
docker ps | grep redis

# View logs
docker logs redis

# Follow logs in real-time
docker logs -f redis

# Interactive Redis CLI
docker exec -it redis redis-cli

# Check memory usage
docker exec -it redis redis-cli INFO memory
```

If you need to stop/start:
```bash
docker stop redis
docker start redis
```

Or remove completely:
```bash
docker rm -f redis
docker volume rm redis_data  # if you want to delete data
```""")
        ],
        "score": 7,
        "insights": ["redis:alpine is lightweight", "Use volumes for persistence", "--appendonly yes enables AOF persistence"]
    },

    # Technical - Error Debugging
    {
        "domain": "Technical",
        "tags": ["python", "debugging", "cors", "flask", "error"],
        "title": "Fixing CORS Errors in Flask API",
        "messages": [
            ("user", """I'm getting this error in my browser console when calling my Flask API:

```
Access to fetch at 'http://localhost:5000/api/data' from origin 'http://localhost:3000'
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present
```

How do I fix this?"""),
            ("assistant", """This is a Cross-Origin Resource Sharing (CORS) error. Your frontend (port 3000) and backend (port 5000) are on different origins.

**Quick fix with flask-cors:**

```bash
pip install flask-cors
```

```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Or for specific origins:
CORS(app, origins=['http://localhost:3000'])
```

**Manual fix (if you can't use flask-cors):**

```python
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE'
    return response
```

The key is the `Access-Control-Allow-Origin` header that tells browsers the cross-origin request is allowed."""),
            ("user", "That fixed it! But now I'm getting errors on POST requests with JSON."),
            ("assistant", """For POST/PUT with JSON, you also need to handle preflight OPTIONS requests. Update your CORS config:

```python
CORS(app,
     origins=['http://localhost:3000'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])
```

Or if doing it manually, add OPTIONS handling:

```python
@app.route('/api/data', methods=['POST', 'OPTIONS'])
def api_data():
    if request.method == 'OPTIONS':
        return '', 204  # Preflight response
    # ... your POST logic
```

The browser sends an OPTIONS preflight request before POST/PUT with custom headers to check if it's allowed.""")
        ],
        "score": 8,
        "insights": ["CORS blocks cross-origin requests by default", "flask-cors is the easiest solution", "Preflight OPTIONS needed for POST with JSON"]
    },

    # Technical - Git
    {
        "domain": "Technical",
        "tags": ["git", "version-control", "rebase", "workflow"],
        "title": "Git Rebase vs Merge Strategy",
        "messages": [
            ("user", "Should I use git rebase or git merge to update my feature branch? My team disagrees on this."),
            ("assistant", """Both have their place. Here's when to use each:

**Use `git rebase` when:**
- Working on a local feature branch not yet pushed
- You want a clean, linear history
- Preparing a PR for review

```bash
git checkout feature-branch
git rebase main
# Resolve any conflicts, then:
git push --force-with-lease  # Only if already pushed
```

**Use `git merge` when:**
- The branch is shared/public
- You want to preserve branch history
- Avoiding force-push requirements

```bash
git checkout feature-branch
git merge main
```

**My recommendation:** Rebase for local cleanup before PR, merge for integrating shared branches.

**Golden rule:** Never rebase commits that have been pushed to a shared branch that others might have pulled."""),
            ("user", "What's the difference between --force and --force-with-lease?"),
            ("assistant", """`--force-with-lease` is safer. Here's why:

**`git push --force`:**
- Overwrites remote branch unconditionally
- Can destroy commits others pushed
- Dangerous on shared branches

**`git push --force-with-lease`:**
- Only forces if your local ref matches remote
- Fails if someone else pushed meanwhile
- Prevents accidentally overwriting others' work

```bash
# Safe force push
git push --force-with-lease origin feature-branch

# If it fails, someone else pushed. Fetch and rebase again:
git fetch origin
git rebase origin/feature-branch
git push --force-with-lease
```

Always use `--force-with-lease` instead of `--force`. Some teams even alias it:

```bash
git config --global alias.pushf 'push --force-with-lease'
```""")
        ],
        "score": 7,
        "insights": ["Rebase for local branches, merge for shared", "Never rebase pushed shared branches", "--force-with-lease is safer than --force"]
    },

    # Creative - Writing
    {
        "domain": "Creative",
        "tags": ["writing", "storytelling", "fiction", "character"],
        "title": "Developing Complex Antagonists",
        "messages": [
            ("user", "I'm writing a fantasy novel and my villain feels flat. How do I make them more compelling?"),
            ("assistant", """The best antagonists believe they're the hero of their own story. Here are key techniques:

**1. Give them a valid worldview**
They should have a philosophy that, from their perspective, justifies their actions. Thanos wanted balance. Killmonger wanted justice for his people.

**2. Show their origin**
What made them this way? A tragic backstory creates empathy without excusing behavior.

**3. Make them competent**
They should win sometimes. A villain who always loses isn't threatening.

**4. Give them relationships**
Do they love someone? Have loyal followers? Care about something beyond their goal?

**5. Mirror the protagonist**
The best villains reflect a dark possible path the hero could have taken.

What's your villain's core motivation? That'll help us develop them further."""),
            ("user", "She wants to prevent a prophecy that says her kingdom will fall. She's willing to do terrible things to stop it."),
            ("assistant", """That's a fantastic foundation! She's essentially trying to protect her people - a noble goal with dark methods. Here's how to deepen her:

**The Tragic Irony:** Make her actions the very thing that causes the prophecy to fulfill. Her cruelty creates the enemies who destroy her.

**Humanizing moments:**
- Show her genuinely caring for her subjects' welfare
- A scene where she hesitates before a terrible act
- Someone she loves who doesn't know her dark side

**Complexity in methods:**
- Maybe she tried peaceful solutions first and they failed
- She believes she's sacrificing her own soul to save thousands

**The question she forces on readers:** How far would YOU go to protect everyone you love?

**Character detail:** Give her a small, humanizing habit - she tends a garden, writes poetry, or has a beloved pet. Contrast with her ruthlessness.

This creates a villain readers might hate, fear, but also understand - maybe even sympathize with in moments.""")
        ],
        "score": 9,
        "insights": ["Villains should believe they're the hero", "Mirror the protagonist's potential dark path", "Tragic irony: their actions cause the outcome they fear"]
    },

    # Business - Strategy
    {
        "domain": "Business",
        "tags": ["startup", "pricing", "strategy", "saas"],
        "title": "SaaS Pricing Strategy for Early Stage",
        "messages": [
            ("user", "We're launching a B2B SaaS product. How should we think about pricing? We have no idea what to charge."),
            ("assistant", """Pricing is one of the highest-leverage decisions you'll make. Here's a framework:

**1. Start with value-based pricing**
- What problem do you solve?
- What's the cost of NOT solving it?
- What would customers pay to make that problem disappear?

**2. Research competitors**
- Not to copy, but to understand the market
- Position yourself: cheaper, premium, or different?

**3. Talk to potential customers**
- "If this solved X, what would that be worth?"
- "What are you paying now for alternatives?"

**4. Start higher than you think**
- It's easier to lower prices than raise them
- Early customers accept higher prices for access/influence
- You can always add a cheaper tier later

**5. Simple tiers to start**
- 3 tiers maximum
- Clear feature differentiation
- One "recommended" tier at your target price

What problem does your product solve, and who's your target customer?"""),
            ("user", "We help sales teams track and follow up on leads automatically. Target is companies with 10-50 salespeople."),
            ("assistant", """Perfect, let's get specific:

**Value calculation:**
- Average salesperson: $60-80k salary
- If your tool saves 5 hours/week = ~$7k/year value per rep
- For a 20-person team = $140k annual value
- You could reasonably charge 10-20% of value created

**Suggested pricing structure:**

| Tier | Users | Price/mo | Per user |
|------|-------|----------|----------|
| Starter | Up to 10 | $299 | $30 |
| Growth | Up to 25 | $599 | $24 |
| Scale | Up to 50 | $999 | $20 |
| Enterprise | 50+ | Custom | Varies |

**Why this works:**
- Volume discount incentivizes growth
- $299 entry is low enough for SMB budget approval
- Enterprise tier captures larger deals

**Pricing page tips:**
- Highlight Growth as "Most Popular"
- Annual discount (20%) to improve cash flow
- Free trial, not freemium (you want qualified leads)

Start here and iterate based on conversion data and customer feedback.""")
        ],
        "score": 8,
        "insights": ["Price based on value created, not cost", "Start higher and adjust down if needed", "3 tiers maximum for simplicity"]
    },

    # Technical - SQL
    {
        "domain": "Technical",
        "tags": ["sql", "database", "postgresql", "optimization"],
        "title": "Optimizing Slow SQL Queries",
        "messages": [
            ("user", """This query takes 30 seconds on our production database:

```sql
SELECT u.*, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id
ORDER BY order_count DESC
LIMIT 100;
```

How can I speed it up?"""),
            ("assistant", """Let's diagnose and optimize. First, run EXPLAIN ANALYZE:

```sql
EXPLAIN ANALYZE SELECT u.*, ...
```

**Likely issues and fixes:**

**1. Missing indexes** (most common)
```sql
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

**2. SELECT * is wasteful**
Only select columns you need:
```sql
SELECT u.id, u.name, u.email, COUNT(o.id) as order_count
```

**3. Consider a covering index**
If you frequently query this pattern:
```sql
CREATE INDEX idx_orders_user_id_covering ON orders(user_id) INCLUDE (id);
```

**4. For very large tables, materialized view**
```sql
CREATE MATERIALIZED VIEW user_order_counts AS
SELECT user_id, COUNT(*) as order_count
FROM orders
GROUP BY user_id;

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY user_order_counts;
```

Run EXPLAIN ANALYZE before and after changes to measure improvement. What does the current plan show?"""),
            ("user", "The EXPLAIN shows a Seq Scan on orders (15 million rows). Adding the index dropped it to 2 seconds!"),
            ("assistant", """The index made the difference - you went from a sequential scan (reading every row) to an index scan.

**For further optimization:**

1. **Partial index** if you only query recent orders:
```sql
CREATE INDEX idx_orders_recent ON orders(user_id)
WHERE created_at > '2024-01-01';
```

2. **Statistics update** after big data changes:
```sql
ANALYZE orders;
ANALYZE users;
```

3. **Monitor with pg_stat_statements:**
```sql
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

**Performance checklist for production:**
- Index foreign keys (user_id)
- Index columns in WHERE clauses
- Index columns in ORDER BY
- Run ANALYZE after bulk loads
- Monitor slow query log

2 seconds is good, but if you need sub-100ms, the materialized view approach is your next step.""")
        ],
        "score": 9,
        "insights": ["Missing indexes cause most slow queries", "Use EXPLAIN ANALYZE to diagnose", "Materialized views for complex aggregations"]
    },

    # Technical - React
    {
        "domain": "Technical",
        "tags": ["react", "javascript", "hooks", "performance"],
        "title": "React useEffect Cleanup and Memory Leaks",
        "messages": [
            ("user", """I'm getting this warning in React:
"Can't perform a React state update on an unmounted component"

What's causing it and how do I fix it?"""),
            ("assistant", """This happens when an async operation completes after the component unmounts. Common causes:

**1. Fetch without cleanup:**
```jsx
// BAD
useEffect(() => {
  fetch('/api/data')
    .then(res => res.json())
    .then(data => setData(data)); // Component might be unmounted!
}, []);

// GOOD
useEffect(() => {
  let isMounted = true;

  fetch('/api/data')
    .then(res => res.json())
    .then(data => {
      if (isMounted) setData(data);
    });

  return () => { isMounted = false; };
}, []);
```

**2. With AbortController (better):**
```jsx
useEffect(() => {
  const controller = new AbortController();

  fetch('/api/data', { signal: controller.signal })
    .then(res => res.json())
    .then(data => setData(data))
    .catch(err => {
      if (err.name !== 'AbortError') throw err;
    });

  return () => controller.abort();
}, []);
```

**3. Timers/intervals:**
```jsx
useEffect(() => {
  const timer = setTimeout(() => setData('updated'), 5000);
  return () => clearTimeout(timer);
}, []);
```

The cleanup function (returned from useEffect) runs when the component unmounts."""),
            ("user", "What about with async/await syntax?"),
            ("assistant", """You can't make useEffect's callback async directly, but here's the pattern:

```jsx
useEffect(() => {
  const controller = new AbortController();

  const fetchData = async () => {
    try {
      const response = await fetch('/api/data', {
        signal: controller.signal
      });
      const data = await response.json();
      setData(data);
    } catch (error) {
      if (error.name !== 'AbortError') {
        setError(error);
      }
    }
  };

  fetchData();

  return () => controller.abort();
}, []);
```

**Or create a custom hook for reuse:**

```jsx
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);

    fetch(url, { signal: controller.signal })
      .then(res => res.json())
      .then(setData)
      .catch(err => {
        if (err.name !== 'AbortError') setError(err);
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [url]);

  return { data, loading, error };
}

// Usage
const { data, loading, error } = useFetch('/api/data');
```

This pattern handles all the edge cases cleanly.""")
        ],
        "score": 8,
        "insights": ["Always cleanup async operations in useEffect", "AbortController is the modern way to cancel fetch", "Custom hooks encapsulate reusable async logic"]
    },

    # Personal - Productivity
    {
        "domain": "Personal",
        "tags": ["productivity", "habits", "focus", "time-management"],
        "title": "Building a Sustainable Productivity System",
        "messages": [
            ("user", "I keep trying different productivity systems (GTD, Pomodoro, etc.) but nothing sticks. How do I find something sustainable?"),
            ("assistant", """The issue isn't the system - it's trying to adopt someone else's complete system. Here's a better approach:

**Start with principles, not systems:**

1. **Capture everything** - Get tasks out of your head
2. **Review regularly** - Weekly minimum
3. **Single focus** - One thing at a time
4. **Energy management** - Match task difficulty to energy levels

**Build YOUR system incrementally:**

Week 1: Just capture
- One place for all tasks (app, notebook, whatever)
- Don't organize, just capture

Week 2: Add daily review
- Each morning: pick 3 things for today
- Each evening: capture tomorrow's thoughts

Week 3: Add weekly review
- Sunday: review all captured items
- Delete, delegate, schedule, or do

**Why this works:**
- You're building habits, not adopting a system
- Each addition is small and sustainable
- The system evolves to fit your life

What's your biggest struggle currently - capturing, deciding, or doing?"""),
            ("user", "Definitely deciding. I have too many things and don't know what's actually important."),
            ("assistant", """Deciding is about clarity on what matters. Try the Eisenhower Matrix, but simpler:

**Two questions for each task:**

1. "If I could only do ONE thing today, would this be it?"
2. "What happens if I don't do this at all?"

**Daily practice:**

Each morning, from your full list:
- Pick the ONE thing that would make today a win
- Do it first, before email/messages
- Everything else is bonus

**Weekly practice:**

Answer: "What are my 3 priorities for the next 90 days?"
- These become your filter for decisions
- Say no to anything that doesn't advance these

**The uncomfortable truth:**

You probably have 50 tasks but only 3 actually matter. The rest are:
- Things you should delegate or automate
- Things that feel urgent but aren't important
- Things you're avoiding deciding about

**Try this now:** Look at your task list. For each item ask: "What's the worst that happens if I delete this?" Most things, nothing bad happens.

Productivity isn't doing more things. It's doing the right things.""")
        ],
        "score": 8,
        "insights": ["Build habits incrementally, not systems wholesale", "Two key questions: Is this THE priority? What if I don't do it?", "Productivity is doing right things, not more things"]
    },

    # Technical - Testing
    {
        "domain": "Technical",
        "tags": ["testing", "python", "pytest", "mocking"],
        "title": "Effective Unit Testing with Pytest",
        "messages": [
            ("user", "How do I test a function that makes API calls? I don't want to hit the real API in tests."),
            ("assistant", """Use mocking to replace the API call. Here's the pattern with pytest:

**Your code (api_client.py):**
```python
import requests

def get_user(user_id):
    response = requests.get(f'https://api.example.com/users/{user_id}')
    response.raise_for_status()
    return response.json()
```

**Your test (test_api_client.py):**
```python
from unittest.mock import patch, Mock
import pytest
from api_client import get_user

def test_get_user_success():
    mock_response = Mock()
    mock_response.json.return_value = {'id': 1, 'name': 'Alice'}
    mock_response.raise_for_status = Mock()

    with patch('api_client.requests.get', return_value=mock_response):
        result = get_user(1)

    assert result == {'id': 1, 'name': 'Alice'}

def test_get_user_not_found():
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError('404')

    with patch('api_client.requests.get', return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            get_user(999)
```

**Key points:**
- Patch where it's used, not where it's defined
- Mock the response object, not just the function
- Test both success and error paths"""),
            ("user", "What about testing async functions?"),
            ("assistant", """For async, use `pytest-asyncio` and `AsyncMock`:

```bash
pip install pytest-asyncio
```

**Your async code:**
```python
import aiohttp

async def fetch_user(session, user_id):
    async with session.get(f'https://api.example.com/users/{user_id}') as response:
        return await response.json()
```

**Your test:**
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_fetch_user():
    # Create mock response
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={'id': 1, 'name': 'Alice'})

    # Create mock context manager
    mock_session = MagicMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    result = await fetch_user(mock_session, 1)

    assert result == {'id': 1, 'name': 'Alice'}
    mock_session.get.assert_called_once_with('https://api.example.com/users/1')
```

**Fixture for reusable mock session:**
```python
@pytest.fixture
def mock_session():
    session = MagicMock()
    response = MagicMock()
    response.json = AsyncMock()
    session.get.return_value.__aenter__.return_value = response
    return session, response

@pytest.mark.asyncio
async def test_fetch_user_with_fixture(mock_session):
    session, response = mock_session
    response.json.return_value = {'id': 1, 'name': 'Alice'}

    result = await fetch_user(session, 1)
    assert result['name'] == 'Alice'
```

The key is `AsyncMock` for async methods and proper `__aenter__` mocking for context managers.""")
        ],
        "score": 8,
        "insights": ["Patch where used, not where defined", "Mock response objects, not just functions", "Use AsyncMock for async methods"]
    },
]


# =============================================================================
# Generator Functions
# =============================================================================

def generate_conversation(template: Dict, index: int, source: str = None) -> Dict[str, Any]:
    """Generate an enriched conversation from a template."""
    sources = ["OpenAI", "Anthropic", "Google"]
    source = source or random.choice(sources)

    # Generate timestamps
    base_date = datetime.now() - timedelta(days=random.randint(1, 365))
    create_time = base_date - timedelta(hours=random.randint(0, 23))

    # Build raw text
    raw_text_parts = [f"Title: {template['title']}\n"]
    for role, content in template['messages']:
        role_upper = "USER" if role == "user" else "ASSISTANT"
        raw_text_parts.append(f"\n{role_upper}: {content}\n")

    raw_text = "".join(raw_text_parts)

    return {
        "convo_id": str(uuid.uuid4()),
        "external_id": f"{source.lower()}-{uuid.uuid4().hex[:8]}",
        "timestamp": create_time.isoformat(),
        "source": source,
        "raw_text": raw_text,
        "generated_title": template["title"],
        "summary_abstractive": f"A conversation about {template['domain'].lower()} topics covering {', '.join(template['tags'][:3])}.",
        "summary_extractive": template['insights'][0] if template.get('insights') else "",
        "primary_domain": template["domain"],
        "tags": template["tags"],
        "key_topics": template["tags"][:3],
        "brilliance_score": {
            "score": template.get("score", 7),
            "reasoning": f"Good {template['domain'].lower()} discussion with practical insights",
            "factors": {
                "depth": random.randint(6, 9),
                "actionability": random.randint(6, 9),
                "creativity": random.randint(5, 8),
                "problem_solving": random.randint(6, 9)
            }
        },
        "key_insights": template.get("insights", []),
        "status": "Completed",
        "future_potential": {
            "value_proposition": f"Reference for {template['tags'][0]}",
            "next_steps": ""
        },
        "score": template.get("score", 7),
        "score_reasoning": f"Practical {template['domain'].lower()} content with clear examples",
        "metadata": {
            "message_count": len(template["messages"]) * 2,
            "user_messages": len(template["messages"]),
            "assistant_messages": len(template["messages"])
        }
    }


def generate_sample_dataset(count: int = 100) -> List[Dict[str, Any]]:
    """Generate a dataset of sample conversations."""
    conversations = []

    # Generate variations of each template
    templates = CONVERSATIONS * ((count // len(CONVERSATIONS)) + 1)

    for i, template in enumerate(templates[:count]):
        # Vary the source distribution
        if i % 3 == 0:
            source = "OpenAI"
        elif i % 3 == 1:
            source = "Anthropic"
        else:
            source = "Google"

        conv = generate_conversation(template, i, source)
        conversations.append(conv)

    return conversations


def save_dataset(conversations: List[Dict], output_path: str):
    """Save conversations to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"Saved {len(conversations)} conversations to {path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample conversation data")
    parser.add_argument('--count', type=int, default=100, help="Number of conversations to generate")
    parser.add_argument('--output', type=str, default='data/enriched_repository.jsonl',
                        help="Output file path")

    args = parser.parse_args()

    print(f"Generating {args.count} sample conversations...")
    conversations = generate_sample_dataset(args.count)
    save_dataset(conversations, args.output)
    print("Done!")
