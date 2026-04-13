"""
Chatbot read-only tools.

Pure async functions that wrap existing CRUD and service calls for use as
LangGraph tool nodes. Every function here is strictly read-only: no writes
to user, movie, swipe, watchlist, or preference data.

These functions return structured Python dicts (not prose) sized for
compact LLM context. Cast lists, trailer URLs, image URLs, and provider
details are intentionally omitted to keep token usage low.
"""

import logging
from typing import Any

from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.interactions import get_user_liked_movies
from movie_recommender.database.CRUD.movies import (
    get_filtered_movie_ids,
    movies_to_details_bulk,
)
from movie_recommender.database.CRUD.users import (
    get_user_analytics,
    get_user_excluded_genres,
    get_user_included_genres,
    get_user_preferences,
)
from movie_recommender.services.knowledge_graph.beacon import get_top_people

logger = logging.getLogger(__name__)

# Maximum IDs to hydrate in a single search call.
# get_filtered_movie_ids can return thousands of rows; cap hydration
# to keep DB round-trips and LLM context bounded.
_SEARCH_HYDRATE_CAP = 15

# Maximum liked movies included in the taste summary.
_TASTE_LIKED_MOVIE_CAP = 10


async def search_movies(
    db: AsyncSession,
    genre_names: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the local movie catalogue by genre and/or release year.

    Filters against the hydrated movie index (movies with a TMDB ID and poster).
    There is no full-text or keyword search — the LLM should use the synopsis
    fields in the results to reason about semantic criteria (e.g. "twist ending",
    "based on a true story").

    A movie matches genre_names if it has AT LEAST ONE of the supplied genres,
    matching the existing feed-manager behaviour in get_filtered_movie_ids.

    Args:
        db: Active async database session.
        genre_names: Genre names to filter by (e.g. ["Thriller", "Mystery"]).
                     Pass None or [] to skip genre filtering.
        min_year: Lower bound on release year (inclusive). None means no lower bound.
        max_year: Upper bound on release year (inclusive). None means no upper bound.
        limit: Maximum results to return. Hard-capped at _SEARCH_HYDRATE_CAP.

    Returns:
        List of compact movie dicts ordered by the DB's natural row order.
        Returns [] on error or when no movies match the filters.
    """
    effective_limit = min(limit, _SEARCH_HYDRATE_CAP)

    try:
        movie_ids = await get_filtered_movie_ids(
            db,
            genre_names=genre_names or None,
            min_year=min_year,
            max_year=max_year,
        )
    except Exception:
        logger.warning("search_movies: get_filtered_movie_ids failed", exc_info=True)
        return []

    if not movie_ids:
        return []

    # get_filtered_movie_ids returns an unordered set; slice before hydrating
    # to avoid fetching thousands of rows when the filter is broad.
    ids_to_fetch = list(movie_ids)[:effective_limit]

    try:
        details = await movies_to_details_bulk(db, ids_to_fetch)
    except Exception:
        logger.warning("search_movies: movies_to_details_bulk failed", exc_info=True)
        return []

    return [_movie_to_compact(m) for m in details]


async def get_user_taste_summary(
    db: AsyncSession,
    redis_client,
    neo4j_driver: AsyncDriver,
    user_id: int,
) -> dict[str, Any]:
    """
    Build a structured taste profile for a user from their swipe history.

    Combines four data sources:
    - Swipe analytics: total counts and top liked genres (PostgreSQL).
    - Explicit preferences: year range, rating floor, genre allow/block lists (PostgreSQL).
    - Top liked movies: up to _TASTE_LIKED_MOVIE_CAP highest-scored movies with
      title, year, genres, and synopsis (PostgreSQL).
    - Top people: highest-weight directors, actors, and writers derived from the
      knowledge graph beacon map (Redis cache + Neo4j on miss).

    Each section degrades gracefully: a failure in one source leaves the others
    intact and logs a warning rather than raising.

    Note on read-only contract: get_top_people writes to Redis on a beacon cache
    miss. This is a cache populate, not a mutation of user preference or swipe data,
    and is consistent with how the existing /users/{user_id}/top-people endpoint
    behaves.

    Args:
        db: Active async database session.
        redis_client: Async Redis client used by the beacon map cache.
        neo4j_driver: Neo4j async driver for knowledge graph queries.
        user_id: Internal integer user ID (not firebase_uid).

    Returns:
        Dict with keys: stats, preferences, top_liked_movies, top_people.
        Any section that fails returns an empty dict or list for that key.
    """
    summary: dict[str, Any] = {}

    # 1. Swipe analytics — total counts and top liked genres
    try:
        analytics = await get_user_analytics(db, user_id)
        summary["stats"] = {
            "total_likes": analytics["total_likes"],
            "total_dislikes": analytics["total_dislikes"],
            "total_seen": analytics["total_seen"],
            "top_genres": analytics["top_genres"],
        }
    except Exception:
        logger.warning(
            "get_user_taste_summary: analytics failed for user_id=%s",
            user_id,
            exc_info=True,
        )
        summary["stats"] = {}

    # 2. Explicit preferences — year range, rating floor, genre allow/block lists
    try:
        prefs = await get_user_preferences(db, user_id)
        included = await get_user_included_genres(db, user_id)
        excluded = await get_user_excluded_genres(db, user_id)
        summary["preferences"] = {
            "min_release_year": prefs.min_year if prefs else None,
            "max_release_year": prefs.max_year if prefs else None,
            "min_rating": prefs.min_rating if prefs else None,
            "include_adult": prefs.include_adult if prefs else False,
            "included_genres": included,
            "excluded_genres": excluded,
        }
    except Exception:
        logger.warning(
            "get_user_taste_summary: preferences failed for user_id=%s",
            user_id,
            exc_info=True,
        )
        summary["preferences"] = {}

    # 3. Top liked movies — highest preference-score movies with compact details
    try:
        liked_ids, _ = await get_user_liked_movies(
            db, user_id, limit=_TASTE_LIKED_MOVIE_CAP
        )
        if liked_ids:
            liked_details = await movies_to_details_bulk(db, liked_ids)
            summary["top_liked_movies"] = [_movie_to_compact(m) for m in liked_details]
        else:
            summary["top_liked_movies"] = []
    except Exception:
        logger.warning(
            "get_user_taste_summary: liked movies failed for user_id=%s",
            user_id,
            exc_info=True,
        )
        summary["top_liked_movies"] = []

    # 4. Top people — knowledge-graph-derived directors, actors, writers
    try:
        top_people = await get_top_people(
            redis_client, neo4j_driver, db, user_id, limit=5
        )
        summary["top_people"] = {
            "directors": [_person_to_compact(p) for p in top_people.get("directors", [])],
            "actors": [_person_to_compact(p) for p in top_people.get("actors", [])],
            "writers": [_person_to_compact(p) for p in top_people.get("writers", [])],
        }
    except Exception:
        logger.warning(
            "get_user_taste_summary: top_people failed for user_id=%s",
            user_id,
            exc_info=True,
        )
        summary["top_people"] = {"directors": [], "actors": [], "writers": []}

    return summary


# ---------------------------------------------------------------------------
# Private serialisation helpers
# ---------------------------------------------------------------------------


def _movie_to_compact(movie: Any) -> dict[str, Any]:
    """
    Reduce a MovieDetails object to a compact dict for LLM context.

    Intentionally omits cast, trailer_url, movie_providers, keywords,
    collection, production_companies, and explanation to minimise token use.
    synopsis is included because the LLM uses it for semantic reasoning when
    there is no keyword search (e.g. detecting themes like "twist ending").
    """
    return {
        "movie_db_id": movie.movie_db_id,
        "title": movie.title,
        "release_year": movie.release_year,
        "rating": movie.rating,
        "genres": movie.genres,
        "synopsis": movie.synopsis,
        "runtime_minutes": movie.runtime,
    }


def _person_to_compact(person: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce a top-person dict to the fields useful for taste reasoning.

    Omits image_url and linked_movies (high-cardinality, not needed by the LLM
    for a taste summary). Weight is kept so the LLM can distinguish strong
    preferences from weak ones.
    """
    return {
        "name": person.get("name"),
        "entity_type": person.get("entity_type"),
        "weight": person.get("weight"),
    }
