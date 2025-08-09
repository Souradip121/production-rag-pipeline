import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# --- 1. CONFIGURE AND INITIALIZE ---
API_KEY = os.getenv("PINECONE_API_KEY")
CLOUD = os.getenv("PINECONE_CLOUD")
REGION = os.getenv("PINECONE_REGION")

if not all([API_KEY, CLOUD, REGION]):
    raise ValueError("PINECONE_API_KEY, PINECONE_CLOUD, or PINECONE_REGION not found in environment variables.")

pc = Pinecone(api_key=API_KEY)
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
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
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