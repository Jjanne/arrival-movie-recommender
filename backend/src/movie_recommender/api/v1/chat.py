import redis
from fastapi import APIRouter, Depends, HTTPException
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
        )
    except Exception:
        raise HTTPException(status_code=502, detail="LLM request failed")

    return ChatResponse(reply=reply)
