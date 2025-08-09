import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# --- 1. CONFIGURE AND INITIALIZE ---
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

pc = Pinecone(api_key=api_key)
print("✅ Pinecone client initialized.")


# --- 2. DEFINE INDEX CONFIGURATION ---
INDEX_NAME = "production-rag-pipeline"
VECTOR_DIMENSION = 1536


# --- 3. CREATE INDEX IF IT DOESN'T EXIST ---
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating it now...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        metric="cosine"  # 'cosine' is standard for text embeddings
    )
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(5)
    print(f"✅ Index '{INDEX_NAME}' created successfully and is ready.")
else:
    print(f"✅ Index '{INDEX_NAME}' already exists.")

# --- 4. GET A HANDLE TO THE INDEX ---
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"✅ Successfully connected to index '{INDEX_NAME}'.")
print(f"Index stats: {stats}")