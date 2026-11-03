
import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings


# Load environment variables
load_dotenv()


class EmbeddingService:

    def __init__(self):

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            ),
            openai_api_version="2024-02-01",
        )

    def generate_embedding(self, text: str) -> list[float]:

        if not text.strip():
            raise ValueError("Text cannot be empty")

        return self.embeddings.embed_query(text)

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            raise ValueError("No texts supplied for embedding generation")

        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty strings")

        return self.embeddings.embed_documents(texts)
