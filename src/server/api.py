"""
This script acts as the "Bridge" for our project. It creates a web service
that allows our frontend dashboard to send questions to the RAG engine
and get answers back in real-time.
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from src.engine.generator import ClaimGenerator
from src.engine.retriever import ClaimRetriever

# Set up the FastAPI app
app = FastAPI(title="CheckPrioritizer: Fact-Checking Assistant")

# Initialize our searcher and talker when the server starts
searcher = ClaimRetriever()
talker = ClaimGenerator()


class UserQuery(BaseModel):
    """
    This defines what a request from the user should look like.
    """

    query: str
    count: Optional[int] = 3  # How many claims to find (defaults to 3)


class ApiResponse(BaseModel):
    """
    This defines what our answer back to the user looks like.
    """

    answer: str
    sources: List[dict]


@app.post("/ask", response_model=ApiResponse)
async def handle_question(request: UserQuery):
    """
    The main endpoint. It takes a question, finds relevant research
    claims, and uses the LLM to write a summarized answer.
    """
    try:
        logger.info(f"New request received: {request.query}")

        # 1. Find the most relevant claims in our vector database
        relevant_docs = searcher.search(request.query, k=request.count)

        # 2. Let the LLM read those claims and write an answer
        final_answer = talker.generate_answer(request.query, relevant_docs)

        # 3. Package the answer and the raw evidence together
        evidence = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in relevant_docs
        ]

        return ApiResponse(answer=final_answer, sources=evidence)

    except Exception as e:
        logger.error(f"Something went wrong while processing the question: {e}")
        raise HTTPException(
            status_code=500,
            detail="The RAG engine encountered an unexpected error while processing your request..",
        )


@app.get("/status")
def check_status():
    """
    A simple health check to make sure the server is running.
    """
    return {"status": "online"}
