"""
Conversation parsers for different platforms.

Supports parsing conversation exports from:
- ChatGPT (OpenAI conversations.json)
- Claude (Anthropic JSON/JSONL exports)
- Gemini (Google JSON/HTML exports)
"""

from .base_parser import ConversationParser
from .chatgpt_parser import ChatGPTParser
from .claude_parser import ClaudeParser
from .gemini_parser import GeminiParser

__all__ = [
    "ConversationParser",
    "ChatGPTParser",
    "ClaudeParser",
    "GeminiParser",
]
