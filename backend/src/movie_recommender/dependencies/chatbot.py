from functools import lru_cache

from movie_recommender.services.chatbot.main import ChatbotService


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService()
