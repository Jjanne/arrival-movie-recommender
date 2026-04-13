from fastapi import APIRouter, Depends, HTTPException

from movie_recommender.dependencies.chatbot import get_chatbot_service
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.chat import ChatRequest, ChatResponse
from movie_recommender.services.chatbot.main import ChatbotService

router = APIRouter(prefix="/chat")


@router.post(path="/")
async def chat(
    body: ChatRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
    auth_user=Depends(verify_user()),
) -> ChatResponse:
    """Send a message to the chatbot and receive a single assistant reply."""
    try:
        reply = await chatbot.generate_basic_response(body.message)
    except Exception:
        raise HTTPException(status_code=502, detail="LLM request failed")
    return ChatResponse(reply=reply)
