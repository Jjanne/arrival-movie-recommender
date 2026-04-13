"""
ChatbotService — entry point for the movie recommendation chatbot.

Initialises the LLM client from AppSettings and exposes a clean async
interface for future LangGraph graph integration and tool binding.
Currently provides a single basic generation method with no tools, no graph,
and no database calls.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from movie_recommender.core.settings.main import AppSettings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful movie recommendation assistant. "
    "You help users discover films they will enjoy based on their taste and requests. "
    "Be concise and specific. When recommending movies, always include the title and release year."
)


class ChatbotService:
    """
    LLM-backed chatbot service for movie recommendations.

    Wraps a ChatOpenAI client pointed at OpenRouter. Intended to be used as a
    singleton (one instance per process) once wired into a FastAPI dependency,
    matching the pattern of Recommender and FeedManager.

    Usage (once the dependency is wired):
        service = ChatbotService()
        response = await service.generate_basic_response("Suggest a 90s thriller")
    """

    def __init__(self) -> None:
        self.settings = AppSettings()

        if not self.settings.llm.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your environment before instantiating ChatbotService."
            )

        self._llm = ChatOpenAI(
            model=self.settings.llm.model,
            api_key=self.settings.llm.api_key,
            base_url=self.settings.llm.base_url,
        )

        logger.info(
            "ChatbotService initialised with model=%s base_url=%s",
            self.settings.llm.model,
            self.settings.llm.base_url,
        )

    async def generate_basic_response(self, message: str) -> str:
        """
        Send a single user message to the LLM and return the text response.

        Uses a fixed system prompt. No conversation history, no tools, no streaming.
        This method exists as a minimal smoke-test entry point; it will be replaced
        by a LangGraph graph invocation once the agent is wired up.

        Args:
            message: The user's raw message string.

        Returns:
            The LLM's response as a plain string.

        Raises:
            Exception: Propagates any LLM API errors to the caller so the router
                       can return an appropriate HTTP response.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]

        logger.debug("Invoking LLM for message: %.80s", message)

        response = await self._llm.ainvoke(messages)
        return str(response.content)
