from dotenv import load_dotenv
import os

load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")
if not CHROMA_PERSIST_DIR:
    raise ValueError("CHROMA_PERSIST_DIR is not set")

# Define constants for the embedding and chat models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-5-mini"

# Define constants for the chunking and tuning parameters
CHUNK_SIZE = 512 # 512 chars / 128 tokens
CHUNK_OVERLAP = 100

# Define constants for the document summary parameters
SUMMARY_MODEL = "gpt-5-mini"
RETRIEVAL_K = 5
SCORE_THRESHOLD = 0.75

# Define the system prompt for the summary model
SUMMARY_SYSTEM_PROMPT = """
You are a helpful assistant. Summarize the following document in 3-5 sentences.
Say what the document is about and its main points. Be concise.
Respond in markdown without code blocks. Use bold for important numbers or terms, a brief document type line then a bulelt list of key details
Do not wrap your response in code fences."""