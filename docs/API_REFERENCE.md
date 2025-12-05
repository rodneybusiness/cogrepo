# CogRepo API Reference

> **Version:** 2.0.0
> **Base URL:** `http://localhost:5001`

---

## Table of Contents

1. [Core API](#core-api)
2. [Search API](#search-api)
3. [Intelligence API](#intelligence-api)
4. [Health & Monitoring](#health--monitoring)
5. [Import API](#import-api)

---

## Core API

### List Conversations

```
GET /api/conversations
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | - | Text search query |
| `source` | string | - | Filter by source (chatgpt, claude, gemini) |
| `limit` | int | 100 | Max results (capped at 1000) |

**Response:**
```json
{
  "conversations": [
    {
      "convo_id": "abc123",
      "generated_title": "Building a REST API with FastAPI",
      "source": "claude",
      "timestamp": "2024-11-15T10:30:00Z",
      "tags": ["python", "fastapi", "api"],
      "score": 85
    }
  ],
  "total": 150
}
```

---

### Get Conversation

```
GET /api/conversation/<convo_id>
```

**Response:**
```json
{
  "convo_id": "abc123",
  "external_id": "original-id-from-export",
  "generated_title": "Building a REST API with FastAPI",
  "summary_abstractive": "Discussion about creating...",
  "raw_text": "Full conversation text...",
  "source": "claude",
  "timestamp": "2024-11-15T10:30:00Z",
  "tags": ["python", "fastapi"],
  "score": 85,
  "has_code": true,
  "code_block_count": 5
}
```

---

### Get Statistics

```
GET /api/stats
```

**Response:**
```json
{
  "total_conversations": 1500,
  "sources": {
    "chatgpt": 800,
    "claude": 600,
    "gemini": 100
  },
  "date_range": {
    "earliest": "2023-01-15",
    "latest": "2024-12-01"
  },
  "avg_score": 72.5,
  "top_tags": [
    ["python", 450],
    ["javascript", 320],
    ["react", 180]
  ]
}
```

---

### Get Tags

```
GET /api/tags
```

**Response:**
```json
{
  "tags": {
    "python": 450,
    "javascript": 320,
    "react": 180,
    "docker": 95
  }
}
```

---

### Get Sources

```
GET /api/sources
```

**Response:**
```json
{
  "sources": {
    "chatgpt": 800,
    "claude": 600,
    "gemini": 100
  }
}
```

---

## Search API

### Full-Text Search

```
GET /api/search
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | - | Search query (required) |
| `source` | string | - | Filter by source |
| `tag` | string | - | Filter by tag (case-insensitive) |
| `date_from` | string | - | Start date (YYYY-MM-DD) |
| `date_to` | string | - | End date (YYYY-MM-DD) |
| `min_score` | float | - | Minimum score filter |
| `page` | int | 1 | Page number |
| `limit` | int | 25 | Results per page (max 1000) |

**Example:**
```
GET /api/search?q=machine+learning&source=claude&min_score=70&page=1&limit=25
```

**Response:**
```json
{
  "results": [
    {
      "convo_id": "ml-123",
      "generated_title": "Training Neural Networks",
      "summary_abstractive": "Deep dive into...",
      "score": 92,
      "tags": ["machine-learning", "pytorch"]
    }
  ],
  "total": 45,
  "page": 1,
  "limit": 25
}
```

---

### v2 Search (Enhanced)

```
GET /api/v2/search
```

Enhanced search with FTS5 and additional features.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | - | Search query |
| `source` | string | - | Source filter |
| `tag` | string | - | Tag filter |
| `has_code` | bool | - | Filter by code presence |
| `min_score` | int | - | Minimum score (1-100) |
| `sort` | string | `relevance` | Sort: relevance, date, score |
| `page` | int | 1 | Page number |
| `limit` | int | 25 | Results per page |

**Response:**
```json
{
  "results": [...],
  "total": 150,
  "page": 1,
  "limit": 25,
  "facets": {
    "sources": {"chatgpt": 80, "claude": 70},
    "tags": {"python": 50, "javascript": 30}
  }
}
```

---

### Search Suggestions

```
GET /api/suggestions
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | - | Partial query (min 2 chars) |
| `limit` | int | 10 | Max suggestions (capped at 50) |

**Response:**
```json
{
  "suggestions": [
    "Building REST APIs with FastAPI",
    "Building GraphQL servers",
    "Building microservices"
  ]
}
```

---

## Intelligence API

All intelligence endpoints are under `/api/intelligence/`.

### Health Check

```
GET /api/intelligence/health
```

Check availability of intelligence features.

**Response:**
```json
{
  "status": "ok",
  "features": {
    "knowledge_graph": "available",
    "recommendations": "available",
    "insights": "available",
    "topic_segmentation": "available",
    "advanced_scoring": "available",
    "clustering": "available",
    "embeddings": "available"
  }
}
```

---

### Knowledge Graph

#### Get Graph Data

```
GET /api/intelligence/knowledge-graph
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `max_nodes` | int | 100 | Maximum nodes to return |
| `max_edges` | int | 200 | Maximum edges to return |
| `min_count` | int | 2 | Minimum entity occurrence |

**Response:**
```json
{
  "nodes": [
    {"id": "python", "type": "technology", "count": 450, "size": 45},
    {"id": "react", "type": "technology", "count": 180, "size": 18}
  ],
  "links": [
    {"source": "python", "target": "fastapi", "weight": 35},
    {"source": "react", "target": "typescript", "weight": 28}
  ]
}
```

#### Get Entity Relationships

```
GET /api/intelligence/knowledge-graph/entity/<entity_name>
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 10 | Max related entities |

**Response:**
```json
{
  "entity": "Python",
  "related": [
    {"name": "FastAPI", "type": "framework", "weight": 0.85, "count": 120},
    {"name": "Django", "type": "framework", "weight": 0.72, "count": 95},
    {"name": "pytest", "type": "tool", "weight": 0.68, "count": 80}
  ]
}
```

#### Get Top Entities

```
GET /api/intelligence/knowledge-graph/top-entities
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | - | Filter by entity type |
| `limit` | int | 20 | Max entities |

**Response:**
```json
{
  "entities": [
    {"name": "Python", "type": "technology", "count": 450},
    {"name": "JavaScript", "type": "technology", "count": 320}
  ],
  "types": ["technology", "framework", "tool", "concept", "library"]
}
```

---

### Recommendations

#### Similar Conversations

```
GET /api/intelligence/recommendations/<convo_id>
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 5 | Max recommendations |

**Response:**
```json
{
  "source": "abc123",
  "recommendations": [
    {
      "conversation_id": "def456",
      "score": 0.92,
      "reason": "similar_content",
      "title": "Related discussion about FastAPI",
      "tags": ["python", "fastapi"]
    }
  ]
}
```

#### Trending Conversations

```
GET /api/intelligence/recommendations/trending
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `days` | int | 7 | Look-back period |
| `limit` | int | 10 | Max results |

**Response:**
```json
{
  "period_days": 7,
  "trending": [
    {
      "conversation_id": "xyz789",
      "score": 0.95,
      "reason": "high_quality_recent",
      "title": "Building AI Applications",
      "tags": ["ai", "python"]
    }
  ]
}
```

---

### Insights

#### Get Insights

```
GET /api/intelligence/insights
```

**Response:**
```json
{
  "insights": [
    {
      "type": "trend",
      "title": "Rising interest in AI/ML",
      "description": "AI-related conversations increased 45% this month",
      "importance": "high",
      "data": {"growth_rate": 0.45}
    }
  ],
  "count": 15
}
```

#### Get Dashboard

```
GET /api/intelligence/insights/dashboard
```

Complete dashboard data for visualization.

**Response:**
```json
{
  "summary": {
    "total_conversations": 1500,
    "avg_score": 72.5,
    "top_domain": "software-development"
  },
  "activity": {
    "2024-11": 120,
    "2024-10": 95,
    "2024-09": 88
  },
  "trends": [...],
  "top_tags": [...],
  "score_distribution": {...}
}
```

#### Get Topic Trends

```
GET /api/intelligence/insights/trends/<topic>
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `period` | string | `month` | Aggregation period |

**Response:**
```json
{
  "topic": "python",
  "period": "month",
  "data_points": [
    {"period": "2024-11", "count": 45, "avg_score": 78},
    {"period": "2024-10", "count": 38, "avg_score": 75}
  ],
  "trend": "rising"
}
```

#### Technology Trends

```
GET /api/intelligence/insights/technology-trends
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `period` | string | `month` | Aggregation period |
| `limit` | int | 20 | Max technologies |

**Response:**
```json
{
  "trends": [
    {
      "technology": "Python",
      "current_count": 45,
      "previous_count": 38,
      "growth_rate": 0.18,
      "trend": "rising"
    }
  ],
  "period": "month"
}
```

---

### Topic Segmentation

#### Get Conversation Topics

```
GET /api/intelligence/topics/<convo_id>
```

Segment a conversation into distinct topics.

**Response:**
```json
{
  "conversation_id": "abc123",
  "topic_count": 3,
  "has_topic_transitions": true,
  "topic_summary": "Docker setup → Python debugging → API design",
  "segments": [
    {
      "topic": "Docker Configuration",
      "subtopics": ["container setup", "networking"],
      "start_turn": 0,
      "end_turn": 5,
      "technologies": ["docker", "docker-compose"],
      "confidence": 0.92
    },
    {
      "topic": "Python Debugging",
      "subtopics": ["error handling", "logging"],
      "start_turn": 6,
      "end_turn": 12,
      "technologies": ["python", "pytest"],
      "confidence": 0.88
    }
  ]
}
```

#### Multi-Topic Conversations

```
GET /api/intelligence/topics/multi-topic
```

Find conversations that cover multiple topics.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 50 | Max results |

**Response:**
```json
{
  "multi_topic_conversations": [
    {
      "conversation_id": "abc123",
      "title": "Docker and Python Development",
      "topic_count": 3,
      "topics": ["Docker Configuration", "Python Debugging", "API Design"],
      "summary": "Docker setup → Python debugging → API design"
    }
  ],
  "count": 25
}
```

---

### Scoring

#### Get Conversation Score

```
GET /api/intelligence/score/<convo_id>
```

Get detailed 1-100 score breakdown.

**Response:**
```json
{
  "conversation_id": "abc123",
  "score": {
    "overall": 85,
    "grade": "B+",
    "percentile": 82,
    "dimensions": {
      "technical_depth": {
        "score": 88,
        "factors": ["advanced_concepts", "code_complexity"]
      },
      "practical_value": {
        "score": 90,
        "factors": ["reusable_code", "complete_solution"]
      },
      "completeness": {
        "score": 82,
        "factors": ["resolved", "examples_provided"]
      },
      "clarity": {
        "score": 78,
        "factors": ["well_structured"]
      },
      "uniqueness": {
        "score": 75,
        "factors": ["novel_approach"]
      }
    }
  }
}
```

#### Top Scored Conversations

```
GET /api/intelligence/scores/top
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 20 | Max results |

**Response:**
```json
{
  "top_conversations": [
    {
      "conversation_id": "xyz789",
      "title": "Building Distributed Systems",
      "score": {
        "overall": 95,
        "grade": "A+",
        "percentile": 99
      }
    }
  ],
  "count": 20
}
```

#### Score Distribution

```
GET /api/intelligence/scores/distribution
```

**Response:**
```json
{
  "distribution": {
    "90-100": 45,
    "80-89": 120,
    "70-79": 280,
    "60-69": 350,
    "50-59": 280,
    "40-49": 180,
    "30-39": 120,
    "20-29": 80,
    "10-19": 35,
    "0-9": 10
  },
  "statistics": {
    "count": 1500,
    "average": 62.5,
    "min": 5,
    "max": 98
  }
}
```

---

### Clustering

```
GET /api/intelligence/clusters
```

Get conversation clusters based on embeddings.

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": "cluster-0",
      "label": "Web Development",
      "size": 150,
      "keywords": ["react", "javascript", "css", "frontend"],
      "representative_ids": ["abc123", "def456"],
      "coherence": 0.85
    }
  ],
  "outlier_count": 25,
  "total_clusters": 12
}
```

---

## Health & Monitoring

### Liveness Probe

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-01T10:30:00Z"
}
```

---

### Readiness Probe

```
GET /ready
```

**Response (healthy):**
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "data_file": "ok"
  }
}
```

**Response (not ready):**
```json
{
  "status": "not_ready",
  "checks": {
    "database": "ok",
    "data_file": "missing"
  }
}
```
Status: 503

---

### Detailed Health

```
GET /health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 2
    },
    "data_file": {
      "status": "healthy",
      "size_mb": 45.2,
      "conversation_count": 1500
    },
    "embeddings": {
      "status": "healthy",
      "count": 1500
    }
  },
  "system": {
    "memory_percent": 45.2,
    "cpu_percent": 12.5,
    "disk_percent": 68.0
  }
}
```

---

### Metrics

```
GET /metrics
```

Prometheus-compatible metrics.

**Response:**
```
# HELP cogrepo_conversations_total Total number of conversations
# TYPE cogrepo_conversations_total gauge
cogrepo_conversations_total 1500

# HELP cogrepo_search_requests_total Total search requests
# TYPE cogrepo_search_requests_total counter
cogrepo_search_requests_total 12450
```

---

## Import API

### Upload File

```
POST /api/upload
```

**Form Data:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Export file (.json, .jsonl) |
| `source` | string | No | Source: auto, chatgpt, claude, gemini |
| `enrich` | bool | No | Enable AI enrichment (default: true) |

**Response:**
```json
{
  "import_id": "uuid-here",
  "status": "queued",
  "message": "Import queued for processing"
}
```

---

### Get Import Status

```
GET /api/imports/<import_id>
```

**Response:**
```json
{
  "id": "uuid-here",
  "filename": "conversations.json",
  "source": "chatgpt",
  "status": "completed",
  "created_time": "2024-12-01T10:30:00Z",
  "start_time": "2024-12-01T10:30:01Z",
  "end_time": "2024-12-01T10:32:15Z",
  "stats": {
    "total_processed": 500,
    "new_added": 450,
    "duplicates_skipped": 50,
    "errors": 0
  }
}
```

---

### List Imports

```
GET /api/imports
```

**Response:**
```json
{
  "imports": [
    {
      "id": "uuid-1",
      "filename": "conversations.json",
      "status": "completed",
      "created_time": "2024-12-01T10:30:00Z"
    }
  ]
}
```

---

## WebSocket Events

Connect to the WebSocket at `ws://localhost:5001` for real-time updates.

### Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | server→client | Connection established |
| `import_progress` | server→client | Import progress update |
| `import_complete` | server→client | Import finished successfully |
| `import_error` | server→client | Import failed |
| `subscribe_import` | client→server | Subscribe to import updates |

### Import Progress Payload

```json
{
  "import_id": "uuid-here",
  "current": 150,
  "total": 500,
  "percentage": 30,
  "status": "processing",
  "message": "Enriching conversations..."
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "error_type",
  "message": "Human-readable message",
  "details": {}
}
```

### Error Types

| Status | Type | Description |
|--------|------|-------------|
| 400 | `validation_error` | Invalid input |
| 404 | `not_found` | Resource not found |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `internal_error` | Server error |
| 503 | `service_unavailable` | Feature unavailable |

---

## Rate Limits

Default rate limits (configurable):

| Endpoint | Limit |
|----------|-------|
| Search | 60/minute |
| Intelligence | 30/minute |
| Upload | 10/minute |

Exceeded limits return 429 with `Retry-After` header.
