"""
Author: Tamasa Patra
Purpose: Configuration management and settings loader
What it does:
Loads environment variables from .env file
Defines Settings class with all configuration parameters
Validates API key presence
Provides centralized access to settings via settings object
Key exports: settings (Settings instance)
Dependencies: pydantic-settings, python-dotenv
Lines: ~40
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ChromaDB
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    chroma_collection_name: str = "coffee_maker_manual"
    
    # RAG Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "15"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    
    class Config:
        env_file = ".env"

settings = Settings()

# Validate API key
if not settings.openai_api_key or settings.openai_api_key == "":
    raise ValueError("OPENAI_API_KEY not set in .env file")
