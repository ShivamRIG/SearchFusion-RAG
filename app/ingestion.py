from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from elasticsearch import Elasticsearch

from config import settings


class DocumentIngestion:

    def __init__(self):

        # Embedding Model
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )

        # Qdrant Client
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        # Elasticsearch Client
        self.es = Elasticsearch(
            settings.ELASTICSEARCH_URL
        )

        # Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        self._create_qdrant_collection()
        self._create_es_index()

        # LangChain Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings,
        )


    def _create_qdrant_collection(self):

        collections = self.qdrant.get_collections().collections

        names = [c.name for c in collections]

        if settings.QDRANT_COLLECTION not in names:

            self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            print("Qdrant collection created.")



    def _create_es_index(self):

        if not self.es.indices.exists(
            index=settings.ELASTICSEARCH_INDEX
        ):

            self.es.indices.create(
                index=settings.ELASTICSEARCH_INDEX
            )

            print(" Elasticsearch index created.")


    def load_documents(self, folder="data/documents"):

        documents = []

        folder = Path(folder)

        for file in folder.iterdir():

            if file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))

            elif file.suffix == ".txt":
                loader = TextLoader(str(file))

            elif file.suffix in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(str(file))

            else:
                print(f"Skipping {file.name}")
                continue

            docs = loader.load()

            documents.extend(docs)

        print(f"Loaded {len(documents)} documents.")

        return documents


    def split_documents(self, documents):

        chunks = self.splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks.")

        return chunks


    def index_qdrant(self, chunks):

        self.vector_store.add_documents(chunks)

        print("Indexed into Qdrant.")


    def index_elasticsearch(self, chunks):

        for i, chunk in enumerate(chunks):

            self.es.index(
                index=settings.ELASTICSEARCH_INDEX,
                id=i,
                document={
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                },
            )

        print("Indexed into Elasticsearch.")


    def ingest(self):

        print("=" * 50)
        print("Starting Document Ingestion")
        print("=" * 50)

        documents = self.load_documents()

        chunks = self.split_documents(documents)

        self.index_qdrant(chunks)

        self.index_elasticsearch(chunks)

        print("=" * 50)
        print(" Ingestion Complete")
        print("=" * 50)


if __name__ == "__main__":

    ingestion = DocumentIngestion()

    ingestion.ingest()