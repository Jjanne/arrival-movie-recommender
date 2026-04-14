"""
ChatbotService — LangGraph-based movie recommendation chatbot.

Uses an explicit StateGraph (not the deprecated create_react_agent prebuilt)
so the graph structure is clear, dependency-free, and easy to extend to
streaming by swapping ainvoke() for astream().

Graph shape:
    START → agent → tools_condition → "tools" → agent  (loop)
                                    → "__end__"          (done)

tools_condition (langgraph.prebuilt) routes based on whether the last
AIMessage contains tool calls — no custom routing logic needed.

Current tools (strictly read-only):
    - search_movies_tool  : genre + year filter over the local movie catalogue
    - taste_summary_tool  : user taste profile from swipe history + knowledge graph

Infrastructure args (db, redis_client, neo4j_driver, user_id) are pre-bound
via closure inside _make_tools() so the LLM only sees domain-level arguments.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.services.chatbot.tools import get_user_taste_summary, search_movies

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful movie recommendation assistant. "
    "You help users discover films they will enjoy based on their taste and requests. "
    "Always call a tool before answering — use search_movies_tool to find movies matching "
    "specific criteria, or taste_summary_tool to understand the user's preferences. "
    "Be concise. When recommending movies, include the title and release year."
)


class ChatbotService:
    """
    LangGraph-backed chatbot service for movie recommendations.

    Owns the LLM client (initialised once at construction) and builds a
    per-request ReAct graph with read-only tools closed over the request's
    infrastructure dependencies. Intended to be used as a singleton via
    get_chatbot_service().
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def invoke(
        self,
        message: str,
        db: AsyncSession,
        redis_client: Any,
        neo4j_driver: AsyncDriver,
        user_id: int,
    ) -> str:
        """
        Run the ReAct graph for a single user turn and return the final reply.

        Args:
            message: The user's message.
            db: Active async database session (request-scoped).
            redis_client: Async Redis client (beacon map cache).
            neo4j_driver: Neo4j async driver (knowledge graph).
            user_id: Internal integer user ID (not firebase_uid).

        Returns:
            The agent's final text response as a plain string.

        Raises:
            Exception: Propagates LLM API or tool errors to the router.
        """
        tools = self._make_tools(db, redis_client, neo4j_driver, user_id)
        graph = self._build_graph(tools)

        logger.debug("Invoking LangGraph agent for message: %.80s", message)

        result = await graph.ainvoke({"messages": [HumanMessage(content=message)]})
        return str(result["messages"][-1].content)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_graph(self, tools: list):
        """
        Compile a ReAct StateGraph for the given bound tools.

        Structure:
            agent_node  calls the LLM (with system prompt prepended)
            tools_node  executes any tool calls the LLM requested
            tools_condition  routes agent output to tools_node or END

        To add streaming later: replace graph.ainvoke() in invoke() with
        graph.astream(..., stream_mode="messages") and yield token chunks.
        """
        llm_with_tools = self._llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        async def agent_node(state: MessagesState) -> dict:
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile()

    def _make_tools(
        self,
        db: AsyncSession,
        redis_client: Any,
        neo4j_driver: AsyncDriver,
        user_id: int,
    ) -> list:
        """
        Build LangChain tool objects with infrastructure args pre-bound via closure.

        Tool docstrings are sent to the LLM as tool descriptions; they must be
        accurate about what the tool does and when to use it.
        Only domain-level args are exposed to the LLM — db, redis, neo4j, user_id
        are invisible to it.
        """

        @tool
        async def search_movies_tool(
            genre_names: list[str] | None = None,
            min_year: int | None = None,
            max_year: int | None = None,
            limit: int = 10,
        ) -> list[dict]:
            """Search the movie catalogue by genre and/or release year.
            Use for requests like 'find me a 90s thriller' or 'suggest sci-fi from the 2000s'.
            genre_names matches movies with at least one of the given genres (e.g. ['Thriller']).
            Results include synopsis — use them to reason about themes the user mentioned
            (e.g. 'twist ending', 'based on a true story') when no keyword search exists."""
            return await search_movies(
                db,
                genre_names=genre_names,
                min_year=min_year,
                max_year=max_year,
                limit=limit,
            )

        @tool
        async def taste_summary_tool() -> dict:
            """Retrieve this user's movie taste profile derived from their swipe history.
            Use when the user asks what kind of movies they like, wants personalised
            recommendations, or asks about their viewing preferences.
            Returns top genres, highest-rated movies, preferred directors/actors/writers,
            and explicit preference settings."""
            return await get_user_taste_summary(db, redis_client, neo4j_driver, user_id)

        return [search_movies_tool, taste_summary_tool]
