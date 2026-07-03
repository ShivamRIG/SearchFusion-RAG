import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Langfuse
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

    # Elasticsearch
    ELASTICSEARCH_URL = os.getenv(
        "ELASTICSEARCH_URL",
        "http://localhost:9200"
    )
    ELASTICSEARCH_INDEX = os.getenv(
        "ELASTICSEARCH_INDEX",
        "documents"
    )

    # OpenAI Models
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    CHAT_MODEL = os.getenv(
        "CHAT_MODEL",
        "gpt-4o-mini"
    )


settings = Settings()