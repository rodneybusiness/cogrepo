"""
SOTA Enrichment API

Flask blueprint providing streaming enrichment endpoints with
preview/approval workflow.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, Response, stream_with_context
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enrichment.sota_enricher import SOTAEnricher, EnrichmentResult

# Initialize blueprint
enrichment_bp = Blueprint('enrichment', __name__, url_prefix='/api/enrich')

# In-memory preview cache (simple alternative to Redis)
# Format: {conversation_id: {"data": {...}, "timestamp": float}}
preview_cache = {}
preview_lock = threading.Lock()
PREVIEW_TTL = 3600  # 1 hour expiry

def cleanup_expired_previews():
    """Remove expired previews from cache. NOTE: Caller must hold preview_lock."""
    now = time.time()
    expired = [
        cid for cid, entry in preview_cache.items()
        if now - entry["timestamp"] > PREVIEW_TTL
    ]
    for cid in expired:
        del preview_cache[cid]


def get_enricher() -> SOTAEnricher:
    """Get SOTAEnricher instance (singleton pattern)."""
    if not hasattr(get_enricher, '_instance'):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        get_enricher._instance = SOTAEnricher(anthropic_key, openai_key)
    return get_enricher._instance


def load_conversation(conversation_id: str) -> Dict[str, Any]:
    """Load a conversation from the repository."""
    repo_file = Path(__file__).parent.parent / "data" / "enriched_repository.jsonl"

    with open(repo_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                if conv.get('convo_id') == conversation_id:
                    return conv

    raise ValueError(f"Conversation not found: {conversation_id}")


def save_conversation(conversation: Dict[str, Any]):
    """Save an enriched conversation back to the repository."""
    repo_file = Path(__file__).parent.parent / "data" / "enriched_repository.jsonl"
    temp_file = repo_file.with_suffix('.jsonl.tmp')

    # Rewrite repository with updated conversation
    conversation_id = conversation['convo_id']
    found = False

    with open(repo_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if line.strip():
                conv = json.loads(line)
                if conv.get('convo_id') == conversation_id:
                    # Replace with enriched version
                    outfile.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                    found = True
                else:
                    outfile.write(line)

    if not found:
        raise ValueError(f"Conversation not found during save: {conversation_id}")

    # Atomic replace
    temp_file.replace(repo_file)


@enrichment_bp.route('/single', methods=['GET', 'POST'])
def enrich_single():
    """
    Enrich a single conversation with streaming updates.

    Request (POST body):
    {
        "conversation_id": "abc123",
        "fields": ["title", "summary", "tags", "embedding"]
    }

    Response: Server-Sent Events stream
    """
    # For POST request, data comes from request body
    if request.method == 'POST':
        data = request.json
        conversation_id = data.get('conversation_id')
        fields = data.get('fields', ['title', 'summary', 'tags', 'embedding'])
    else:
        # For GET (EventSource), params come from query string
        conversation_id = request.args.get('conversation_id')
        fields_str = request.args.get('fields', 'title,summary,tags,embedding')
        fields = [f.strip() for f in fields_str.split(',')]

    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400

    def generate():
        """SSE generator for streaming enrichment."""
        try:
            # Load conversation
            conversation = load_conversation(conversation_id)

            # Create enricher and enrich with streaming
            enricher = get_enricher()

            # Run async enrichment in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def enrich_async():
                results = []
                async for result in enricher.enrich_single_streaming(conversation, fields):
                    # Send SSE event
                    event_data = {
                        "type": "partial" if result.partial else "final",
                        "field": result.field,
                        "value": result.value if result.field != "embedding" else None,  # Don't send embedding in stream
                        "original_value": result.original_value if result.field != "embedding" else None,
                        "confidence": result.confidence,
                        "cost": result.cost,
                        "model": result.model,
                        "partial": result.partial
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

                    # Store final results
                    if not result.partial:
                        results.append(result)

                # Calculate total cost
                total_cost = sum(r.cost for r in results)

                # Build enriched conversation
                enriched_conv = conversation.copy()
                for result in results:
                    if result.field == "title":
                        enriched_conv["generated_title"] = result.value
                    elif result.field == "summary":
                        enriched_conv["summary_abstractive"] = result.value
                    elif result.field == "tags":
                        enriched_conv["tags"] = result.value
                    elif result.field == "embedding":
                        # Store embedding separately (too large for preview)
                        pass  # Will handle in approve endpoint

                # Store preview in cache
                preview_data = {
                    "conversation": enriched_conv,
                    "results": [r.to_dict() for r in results],
                    "total_cost": total_cost
                }
                with preview_lock:
                    cleanup_expired_previews()
                    preview_cache[conversation_id] = {
                        "data": preview_data,
                        "timestamp": time.time()
                    }

                # Send completion event
                yield f"data: {json.dumps({'type': 'complete', 'total_cost': total_cost})}\n\n"

            # Run async generator
            async_gen = enrich_async()

            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

            loop.close()

        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@enrichment_bp.route('/bulk', methods=['POST'])
def enrich_bulk():
    """
    Enrich multiple conversations with progress streaming.

    Request:
    {
        "conversation_ids": ["abc123", "def456", ...],
        "fields": ["title", "summary", "tags", "embedding"]
    }

    Response: Server-Sent Events stream
    """
    data = request.json
    conversation_ids = data.get('conversation_ids', [])
    fields = data.get('fields', ['title', 'summary', 'tags', 'embedding'])

    if not conversation_ids:
        return jsonify({"error": "conversation_ids required"}), 400

    def generate():
        """SSE generator for bulk enrichment."""
        try:
            enricher = get_enricher()
            total = len(conversation_ids)
            total_cost = 0.0

            for idx, conversation_id in enumerate(conversation_ids, 1):
                try:
                    # Progress update
                    progress_data = {
                        "type": "progress",
                        "current": idx,
                        "total": total,
                        "conversation_id": conversation_id,
                        "percent": (idx / total) * 100
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

                    # Load and enrich
                    conversation = load_conversation(conversation_id)

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(
                        enricher.enrich_single(conversation, fields)
                    )

                    loop.close()

                    # Update conversation
                    enriched_data = result['enriched_data']
                    conv_cost = result['total_cost']
                    total_cost += conv_cost

                    # Apply enrichments
                    if 'title' in enriched_data:
                        conversation['generated_title'] = enriched_data['title']
                    if 'summary' in enriched_data:
                        conversation['summary_abstractive'] = enriched_data['summary']
                    if 'tags' in enriched_data:
                        conversation['tags'] = enriched_data['tags']

                    # Save immediately
                    save_conversation(conversation)

                    # Send completion for this conversation
                    complete_data = {
                        "type": "conversation_complete",
                        "conversation_id": conversation_id,
                        "cost": conv_cost
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"

                except Exception as e:
                    error_data = {
                        "type": "conversation_error",
                        "conversation_id": conversation_id,
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"

            # Final completion
            final_data = {
                "type": "bulk_complete",
                "total_processed": total,
                "total_cost": total_cost
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@enrichment_bp.route('/preview/<conversation_id>', methods=['GET'])
def get_preview(conversation_id: str):
    """
    Get preview data for a conversation.

    Response:
    {
        "conversation": {...},
        "results": [...],
        "total_cost": 0.05
    }
    """
    with preview_lock:
        cleanup_expired_previews()
        entry = preview_cache.get(conversation_id)

    if not entry:
        return jsonify({"error": "No preview available"}), 404

    return jsonify(entry["data"])


@enrichment_bp.route('/approve', methods=['POST'])
def approve_enrichment():
    """
    Approve and persist an enrichment preview.

    Request:
    {
        "conversation_id": "abc123",
        "edited_values": {  # Optional - if user edited the enriched values
            "title": "Custom Title",
            "summary": "Custom Summary",
            "tags": ["tag1", "tag2"]
        }
    }

    Response:
    {
        "success": true,
        "message": "Enrichment saved"
    }
    """
    data = request.json
    conversation_id = data.get('conversation_id')
    edited_values = data.get('edited_values', {})

    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400

    try:
        # Get preview
        with preview_lock:
            entry = preview_cache.get(conversation_id)

        if not entry:
            return jsonify({"error": "No preview available"}), 404

        preview_data = entry["data"]
        enriched_conv = preview_data['conversation']

        # Apply edited values if provided
        if edited_values:
            if 'title' in edited_values:
                enriched_conv['generated_title'] = edited_values['title']
            if 'summary' in edited_values:
                enriched_conv['summary_abstractive'] = edited_values['summary']
            if 'tags' in edited_values:
                enriched_conv['tags'] = edited_values['tags']

        # Save to repository
        save_conversation(enriched_conv)

        # TODO: Update embeddings if embedding was regenerated
        # This would involve updating data/embeddings.npy and embedding_ids.json

        # Clear preview
        with preview_lock:
            preview_cache.pop(conversation_id, None)

        return jsonify({
            "success": True,
            "message": "Enrichment saved successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enrichment_bp.route('/reject', methods=['POST'])
def reject_enrichment():
    """
    Reject and discard an enrichment preview.

    Request:
    {
        "conversation_id": "abc123"
    }

    Response:
    {
        "success": true,
        "message": "Preview discarded"
    }
    """
    data = request.json
    conversation_id = data.get('conversation_id')

    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400

    # Clear preview from cache
    with preview_lock:
        preview_cache.pop(conversation_id, None)

    return jsonify({
        "success": True,
        "message": "Preview discarded"
    })


@enrichment_bp.route('/estimate', methods=['POST'])
def estimate_cost():
    """
    Estimate cost for enriching conversations.

    Request:
    {
        "conversation_ids": ["abc123", "def456"],
        "fields": ["title", "summary", "tags", "embedding"]
    }

    Response:
    {
        "estimated_cost": 5.25,
        "breakdown": {
            "text_generation": 4.50,
            "embeddings": 0.75
        }
    }
    """
    data = request.json
    conversation_ids = data.get('conversation_ids', [])
    fields = data.get('fields', ['title', 'summary', 'tags', 'embedding'])

    # Rough cost estimates
    TITLE_AVG_COST = 0.0002  # ~$0.0002 per title
    SUMMARY_AVG_COST = 0.0003  # ~$0.0003 per summary
    TAGS_AVG_COST = 0.0003  # ~$0.0003 per tag set
    EMBEDDING_AVG_COST = 0.00005  # ~$0.00005 per embedding

    count = len(conversation_ids)
    text_cost = 0.0
    embedding_cost = 0.0

    if 'title' in fields:
        text_cost += TITLE_AVG_COST * count
    if 'summary' in fields:
        text_cost += SUMMARY_AVG_COST * count
    if 'tags' in fields:
        text_cost += TAGS_AVG_COST * count
    if 'embedding' in fields:
        embedding_cost += EMBEDDING_AVG_COST * count

    return jsonify({
        "estimated_cost": round(text_cost + embedding_cost, 4),
        "breakdown": {
            "text_generation": round(text_cost, 4),
            "embeddings": round(embedding_cost, 4)
        },
        "count": count
    })


@enrichment_bp.route('/health', methods=['GET'])
def health_check():
    """
    Check health of enrichment system.

    Response:
    {
        "status": "healthy",
        "anthropic": true,
        "openai": true,
        "redis": true
    }
    """
    status = {
        "status": "healthy",
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "cache": True  # In-memory cache always available
    }

    # Check if both API keys present
    if not (status["anthropic"] and status["openai"]):
        status["status"] = "unavailable"

    return jsonify(status)
