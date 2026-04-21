"""
Rabbit Community Tools
========================

Real executors for all rabbit_* community tools.
Calls the gateway rabbit endpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")


class _RabbitTool(BaseIntegrationSkill):
    """Base for rabbit community tools."""
    api_key_names = []
    _method: str = "GET"
    _path: str = "/rabbit/posts"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {"x-user-id": user_id}
            async with httpx.AsyncClient(timeout=15.0) as client:
                if self._method == "GET":
                    resp = await client.get(f"{GATEWAY_URL}{self._path}", headers=headers)
                elif self._method == "POST":
                    resp = await client.post(f"{GATEWAY_URL}{self._path}", json={"content": message, "user_id": user_id}, headers=headers)
                elif self._method == "DELETE":
                    resp = await client.delete(f"{GATEWAY_URL}{self._path}", headers=headers)
                else:
                    resp = await client.get(f"{GATEWAY_URL}{self._path}", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return {"success": True, "action": self.skill_id, "summary": str(data)[:3000], "data": data}
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}


class CreateRabbitPostTool(_RabbitTool):
    skill_id = "create_rabbit_post"; skill_name = "Create Post"; _method = "POST"; _path = "/rabbit/posts"; intent_keywords = ["create post", "new post"]

class ListRabbitCommunitiesTool(_RabbitTool):
    skill_id = "list_rabbit_communities"; skill_name = "List Communities"; _path = "/rabbit/communities"; intent_keywords = ["list communities"]

class ListRabbitPostsTool(_RabbitTool):
    skill_id = "list_rabbit_posts"; skill_name = "List Posts"; _path = "/rabbit/posts"; intent_keywords = ["list posts"]

class RabbitVoteTool(_RabbitTool):
    skill_id = "rabbit_vote"; skill_name = "Vote"; _method = "POST"; _path = "/rabbit/vote"; intent_keywords = ["vote post", "upvote", "downvote"]

class CreateRabbitCommunityTool(_RabbitTool):
    skill_id = "create_rabbit_community"; skill_name = "Create Community"; _method = "POST"; _path = "/rabbit/communities"; intent_keywords = ["create community"]

class GetRabbitCommunityTool(_RabbitTool):
    skill_id = "get_rabbit_community"; skill_name = "Get Community"; _path = "/rabbit/communities/detail"; intent_keywords = ["get community"]

class SearchRabbitPostsTool(_RabbitTool):
    skill_id = "search_rabbit_posts"; skill_name = "Search Posts"; _path = "/rabbit/posts/search"; intent_keywords = ["search posts"]

class GetRabbitPostTool(_RabbitTool):
    skill_id = "get_rabbit_post"; skill_name = "Get Post"; _path = "/rabbit/posts/detail"; intent_keywords = ["get post"]

class DeleteRabbitPostTool(_RabbitTool):
    skill_id = "delete_rabbit_post"; skill_name = "Delete Post"; _method = "DELETE"; _path = "/rabbit/posts/detail"; intent_keywords = ["delete post"]

class CreateRabbitCommentTool(_RabbitTool):
    skill_id = "create_rabbit_comment"; skill_name = "Create Comment"; _method = "POST"; _path = "/rabbit/comments"; intent_keywords = ["comment on post"]

class ListRabbitCommentsTool(_RabbitTool):
    skill_id = "list_rabbit_comments"; skill_name = "List Comments"; _path = "/rabbit/comments"; intent_keywords = ["list comments"]

class DeleteRabbitCommentTool(_RabbitTool):
    skill_id = "delete_rabbit_comment"; skill_name = "Delete Comment"; _method = "DELETE"; _path = "/rabbit/comments/detail"; intent_keywords = ["delete comment"]


RABBIT_TOOLS = {
    "create_rabbit_post": CreateRabbitPostTool(),
    "list_rabbit_communities": ListRabbitCommunitiesTool(),
    "list_rabbit_posts": ListRabbitPostsTool(),
    "rabbit_vote": RabbitVoteTool(),
    "create_rabbit_community": CreateRabbitCommunityTool(),
    "get_rabbit_community": GetRabbitCommunityTool(),
    "search_rabbit_posts": SearchRabbitPostsTool(),
    "get_rabbit_post": GetRabbitPostTool(),
    "delete_rabbit_post": DeleteRabbitPostTool(),
    "create_rabbit_comment": CreateRabbitCommentTool(),
    "list_rabbit_comments": ListRabbitCommentsTool(),
    "delete_rabbit_comment": DeleteRabbitCommentTool(),
}
