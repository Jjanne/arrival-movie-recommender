import json
import logging

import redis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.chatbot import get_chatbot_service
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.schemas.requests.chat import ChatRequest, ChatResponse
from movie_recommender.services.chatbot.main import ChatbotService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")


@router.post(path="/")
async def chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_async_redis),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    chatbot: ChatbotService = Depends(get_chatbot_service),
    auth_user=Depends(verify_user()),
) -> ChatResponse:
    """Send a message to the chatbot and receive a single assistant reply."""
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        reply = await chatbot.invoke(
            message=body.message,
            db=db,
            redis_client=redis_client,
            neo4j_driver=neo4j_driver,
            user_id=user.id,
            history=body.history,
        )
    except Exception:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="LLM request failed")

    return ChatResponse(reply=reply)


@router.post(path="/stream")
async def chat_stream(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_async_redis),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    chatbot: ChatbotService = Depends(get_chatbot_service),
    auth_user=Depends(verify_user()),
) -> StreamingResponse:
    """Stream the chatbot reply token-by-token as Server-Sent Events.

    Each event is a JSON object: data: {"token": "..."}\n\n
    The stream is terminated by:                data: [DONE]\n\n
    """
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    async def event_generator():
        try:
            async for token in chatbot.stream(
                message=body.message,
                db=db,
                redis_client=redis_client,
                neo4j_driver=neo4j_driver,
                user_id=user.id,
                history=body.history,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception:
            logger.exception("LLM stream failed")
            yield f"data: {json.dumps({'error': 'LLM request failed'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
