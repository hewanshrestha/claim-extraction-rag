"""
This module manages the data engineering layer for the Claim Extraction RAG.
It handles the ingestion, sanitization, and vectorization of claim detection
and checkworthiness detection datasets, transforming raw TSVs into a persistent
ChromaDB store.
"""

import re
from pathlib import Path
from typing import List

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class ClaimIngestor:
    """
    Orchestrates the data pipeline from raw files to a vectorized knowledge base.
    """

    def __init__(self, data_dir: str):
        """
        Sets up the ingestor with a target directory and initializes logging.
        """
        self.data_dir = Path(data_dir)
        logger.add(
            "logs/ingestion.log", rotation="10 MB", retention="10 days", level="INFO"
        )

    def _validate_tsv(self, file_path: Path) -> bool:
        """
        Internal check to ensure the target path exists and is a valid TSV.
        """
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            return False
        if file_path.suffix != ".tsv":
            logger.warning(f"Unexpected file extension for {file_path.name}")
            return False
        return True

    def _clean_tweet_content(self, text: str) -> str:
        """
        Removes noisy elements common in social media data (URLs, HTML)
        and normalizes spacing to improve embedding quality.
        """
        if not isinstance(text, str):
            return ""

        # Strip links, tags, and collapse multiple spaces into one
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_source_data(self) -> List[pd.DataFrame]:
        """
        Loads the datasets into memory. Handles encoding fallbacks for
        resilient data loading.
        """
        target_files = ["claim_dataset.tsv", "checkworthiness_dataset.tsv"]
        dataframes = []

        for file_name in target_files:
            file_path = self.data_dir / file_name
            if not self._validate_tsv(file_path):
                continue

            logger.info(f"Reading {file_name}...")

            try:
                df = pd.read_csv(
                    file_path, sep="\t", encoding="utf-8", on_bad_lines="warn"
                )
            except UnicodeDecodeError:
                logger.debug(f"UTF-8 failed for {file_name}; falling back to latin-1.")
                df = pd.read_csv(file_path, sep="\t", encoding="latin-1")

            if not df.empty:
                df["source_filename"] = file_name
                dataframes.append(df)
                logger.success(f"Ingested {len(df)} rows from {file_name}")

        return dataframes

    def process_and_unify(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combines disparate datasets and applies the sanitization logic
        to the text column.
        """
        logger.info("Merging datasets and cleaning tweet text...")

        master_df = pd.concat(dataframes, ignore_index=True)
        master_df["tweet_text"] = master_df["tweet_text"].apply(
            self._clean_tweet_content
        )

        logger.success(f"Knowledge base unified with {len(master_df)} records.")
        return master_df

    def build_vector_index(self, df: pd.DataFrame, db_path: str = "db/chroma_db"):
        """
        Converts the cleaned DataFrame into semantic chunks and persists
        them into a local ChromaDB instance.
        """
        # Load documents
        loader = DataFrameLoader(df, page_content_column="tweet_text")
        documents = loader.load()

        # Split for semantic density
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50, add_start_index=True
        )
        chunks = splitter.split_documents(documents)

        logger.info(f"Vectorizing {len(chunks)} text chunks...")

        # Initialize embeddings and store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=db_path
        )

        logger.success(f"Vector store persisted at {db_path}")
        return vector_db


if __name__ == "__main__":
    # Initialize the pipeline
    ingestor = ClaimIngestor(data_dir="data")

    # Run the ingestion workflow
    raw_dfs = ingestor.load_source_data()

    if raw_dfs:
        unified_data = ingestor.process_and_unify(raw_dfs)
        vector_store = ingestor.build_vector_index(unified_data)

        print("\n" + "=" * 45)
        print("KNOWLEDGE BASE READY")
        print("=" * 45)

        # Quick verification of the index
        test_query = "Is there any news about COVID-19 vaccines in Barbados?"
        logger.info(f"Running sanity check for: {test_query}")

        top_matches = vector_store.similarity_search(test_query, k=2)

        for i, match in enumerate(top_matches):
            print(f"\nResult {i + 1}:")
            print(f"Text: {match.page_content[:120]}...")
            print(
                f"Meta: {match.metadata['source_filename']} (ID: {match.metadata['tweet_id']})"
            )
