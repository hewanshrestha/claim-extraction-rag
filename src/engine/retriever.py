"""
This module serves as the search engine.
It handles similarity search, MMR diversity, and metadata filtering.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class ClaimRetriever:
    """
    Interface for querying the local ChromaDB with advanced filtering.
    """

    def __init__(self, db_path: str = "db/chroma_db"):
        """
        Initializes the retriever by loading the existing vector store.
        """
        self.db_path = db_path

        # Must use the SAME embedding model used in ingestion!
        logger.info("Connecting to Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        logger.info(f"Loading Vector Database from {db_path}...")
        self.vector_db = Chroma(
            persist_directory=self.db_path, embedding_function=self.embeddings
        )

    def search(self, query: str, k: int = 4, use_mmr: bool = True):
        """
        Performs a search for relevant claims.

        Args:
            query (str): The user's question.
            k (int): Number of documents to return.
            use_mmr (bool): If True, uses Maximal Marginal Relevance for diversity.
        """
        if use_mmr:
            logger.info(f"Performing MMR search for: '{query}'")
            # fetch_k: Number of docs to initially grab before re-ranking for diversity
            # lambda_mult: 0.5 balances relevance and diversity
            return self.vector_db.max_marginal_relevance_search(
                query, k=k, fetch_k=20, lambda_mult=0.5
            )

        logger.info(f"Performing standard similarity search for: '{query}'")
        return self.vector_db.similarity_search(query, k=k)


if __name__ == "__main__":
    # Test the Retriever independently
    searcher = ClaimRetriever()

    test_query = "What are the latest claims about vaccines in the Caribbean?"
    results = searcher.search(test_query, k=3, use_mmr=True)

    print(f"\n--- Search Results for: {test_query} ---")
    for i, doc in enumerate(results):
        print(f"\n[{i + 1}] {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")
