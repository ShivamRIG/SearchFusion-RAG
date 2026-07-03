from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from reranker import Reranker
from retrieval import HybridRetriever
from generation import Generator
from langfuse_client import LangfuseClient
from config import settings


class HybridRAGPipeline:

    def __init__(self):
        self.reranker = Reranker()
        # LLM
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            temperature=0
        )

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )

        # Qdrant
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

        # Elasticsearch
        self.es = Elasticsearch(
            settings.ELASTICSEARCH_URL
        )

        # Langfuse
        self.langfuse = LangfuseClient()

        # Retriever
        self.retriever = HybridRetriever()

        # Generator
        self.generator = Generator()



    def embed_query(self, question):

        return self.embeddings.embed_query(question)



    def ask(self, question):

        trace = self.langfuse.start_trace(question)

        try:

            # Step 1: Embed
            embedding = self.embed_query(question)

            # Step 2: Retrieve
            docs = self.retriever.retrieve(
                question,
                embedding
            )

            # Step 3: Generate
            answer = self.generator.generate(
                question,
                docs
            )

            trace.update(
                output={
                    "answer": answer
                },
                metadata={
                    "retriever": "hybrid",
                    "documents": len(docs)
                }
            )

            return answer

        except Exception as e:

            trace.update(
                output={
                    "error": str(e)
                }
            )

            raise e


    def stream(self, question):

        trace = self.langfuse.start_trace(question)

        try:

            embedding = self.embed_query(question)

            docs = self.retriever.retrieve(
                question,
                embedding
            )

            prompt = self.generator.create_prompt(
                question,
                docs
            )

            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    yield chunk.content

            trace.update(
                metadata={
                    "streaming": True
                }
            )

        except Exception as e:

            trace.update(
                output={
                    "error": str(e)
                }
            )

            raise e