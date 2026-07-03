from langfuse import Langfuse
from app.config import settings


class LangfuseClient:

    def __init__(self):

        self.client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

    def start_trace(self, question: str):

        return self.client.trace(
            name="Hybrid-RAG",
            input={
                "question": question
            }
        )