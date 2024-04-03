import os
import textwrap
from dotenv import load_dotenv
from llama_index import download_loader
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from llama_index import VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
import re