"""
This module serves as the generation engine.
It uses Groq LLM to synthesize answers based on retrieved claims.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger

load_dotenv()


class ClaimGenerator:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """Initializes the LLM via Groq."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment.")
            raise ValueError("Missing API Key")

        self.llm = ChatGroq(
            temperature=0,  # Keep it deterministic for research
            model_name=model_name,
            groq_api_key=api_key,
        )
        logger.info(f"Generator initialized with model: {model_name}")

    def generate_answer(self, query: str, context: List[Document]) -> str:
        """Synthesizes an answer based on provided context."""

        # 1. Prepare the context string
        context_text = "\n\n".join(
            [
                f"Source [{i + 1}]: {doc.page_content} (Label: {doc.metadata.get('class_label')})"
                for i, doc in enumerate(context)
            ]
        )

        # 2. Define the RAG Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Fact-Checking Research Assistant.
                    Use the following pieces of retrieved claims to answer the user's question.
                    If the claims don't contain the answer, say you don't know.
                    Always cite your sources by [Source Number].

                    Retrieved Claims:
                    {context}""",
                ),
                ("human", "{query}"),
            ]
        )

        # 3. Chain and Invoke
        chain = prompt | self.llm
        logger.info("Generating response from LLM...")
        response = chain.invoke({"context": context_text, "query": query})

        return str(response.content)


if __name__ == "__main__":
    # Test the Generator with a mock context
    from src.engine.retriever import ClaimRetriever

    retriever = ClaimRetriever()
    generator = ClaimGenerator()

    query = "What is the situation with vaccine trials in Cuba?"

    # Run the full RAG cycle locally
    retrieved_docs = retriever.search(query, k=3)
    answer = generator.generate_answer(query, retrieved_docs)

    print(f"\nQUERY: {query}")
    print(f"\nANSWER:\n{answer}")
