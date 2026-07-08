"""
Native LLM Function Calling Tool Definitions
=============================================

OpenAI function-calling format tool schemas for the platform's executable tools.
These are passed to LLMs that support native function calling (OpenAI, Anthropic, Gemini)
so the LLM can decide which tool to invoke based on the user's message.

The neural tool classifier is kept as a fallback for providers without tool support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Tool definitions in OpenAI function-calling format.
# UnifiedLLMClient auto-converts these for Anthropic and Gemini.
NATIVE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, documentation, or any topic the user asks about. Use when the user asks about recent events, needs factual data, or requests real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_generation",
            "description": "Generate an image from a text description using DALL-E. Use when the user asks to create, generate, draw, or design an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_visualizer",
            "description": "Analyze code repositories, generate dependency graphs, run SAST security scans, and create architecture visualizations. Use when the user asks about code analysis, security scanning, dependency graphs, or code architecture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "dependencies", "security", "architecture"],
                        "description": "Type of code analysis to perform",
                    },
                    "target": {
                        "type": "string",
                        "description": "Repository URL, file path, or code snippet to analyze",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_calendar",
            "description": "Manage Google Calendar events: create, list, update, or delete calendar events. Use when the user wants to schedule meetings, check their calendar, add events, or manage appointments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "update", "delete"],
                        "description": "Calendar action to perform",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Event title/summary",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Event start time in ISO 8601 format",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Event end time in ISO 8601 format",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_drive",
            "description": "Access and manage Google Drive files: list, search, upload, download, or share files. Use when the user wants to work with files in Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "search", "upload", "download", "share"],
                        "description": "Drive action to perform",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or file identifier",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search the user's semantic memory (Hash Sphere) for past conversations, stored knowledge, and context. Use when the user asks about previous discussions, wants to recall information, or needs historical context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract the readable content of a specific web page URL. Use when the user gives you a URL, or asks you to look at/read/summarize a specific webpage or article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The exact URL to fetch and extract content from",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_rag_ask",
            "description": "Ask a question directly against the user's stored memory using retrieval-augmented generation. Use for open-ended recall questions ('what do you know about X', 'what have I told you about Y') rather than a literal keyword search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural-language question to answer from the user's memory",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "figma",
            "description": "Access Figma design files: view components, export assets, or inspect design details. Use when the user asks about Figma designs or UI components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["view", "export", "inspect", "list"],
                        "description": "Figma action to perform",
                    },
                    "file_key": {
                        "type": "string",
                        "description": "Figma file key or URL",
                    },
                },
                "required": ["action"],
            },
        },
    },
]

# Map from native tool name to the neural classifier tool ID
NATIVE_TOOL_NAME_TO_CLASSIFIER_ID = {
    "web_search": "web_search",
    "image_generation": "image_generation",
    "code_visualizer": "code_visualizer",
    "google_calendar": "google_calendar",
    "google_drive": "google_drive",
    "memory_search": "memory_search",
    "memory_library": "memory_library",
    "figma": "figma",
    "fetch_url": "fetch_url",
    "memory_rag_ask": "memory_rag_ask",
}


def get_tool_definitions(enabled_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    """Get tool definitions, optionally filtered to enabled tool IDs."""
    if enabled_ids is None:
        return NATIVE_TOOLS
    return [
        t for t in NATIVE_TOOLS
        if t["function"]["name"] in enabled_ids
    ]
