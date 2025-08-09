import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

print("Starting data ingestion process using Azure OpenAI...")
load_dotenv()

# --- 1. Initialize Clients and Define Namespace ---
print("Initializing clients...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "production-rag-pipeline"
NAMESPACE = "main-kb"
index = pc.Index(INDEX_NAME)

# --- 2. Load and Split Documents ---
print("Loading and splitting documents...")
loader = TextLoader("data/sample_document.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Loaded and split {len(docs)} chunks.")

# --- 3. Clear existing vectors in the namespace ---
print(f"Clearing any existing vectors from namespace '{NAMESPACE}'...")
# This will not raise an error if the namespace is empty or does not exist
index.delete(delete_all=True, namespace=NAMESPACE)
print("Namespace cleared.")

# --- 4. Embed and Upsert ---
print(f"Embedding and upserting {len(docs)} chunks to namespace '{NAMESPACE}'...")
doc_id = "doc_chunk_0"
doc = docs[0] # We only have one chunk for this sample

vec = embeddings.embed_query(doc.page_content)
upsert_response = index.upsert(
    vectors=[{"id": doc_id, "values": vec, "metadata": {"text": doc.page_content}}],
    namespace=NAMESPACE
)
print("--- UPSERT RESPONSE ---")
print(upsert_response)

# --- 5. Verify by Fetching ---
print("\n--- VERIFICATION BY FETCHING ---")
# Increase delay to account for indexing latency
print("Waiting 10 seconds for indexing to complete...")
time.sleep(10) 

fetch_response = index.fetch(ids=[doc_id], namespace=NAMESPACE)
print("Fetch response:")
print(fetch_response)

# Corrected the check to use dot notation
if doc_id in fetch_response.vectors:
    print(f"✅ SUCCESS: Vector {doc_id} was successfully fetched from the index.")
else:
    print(f"❌ FAILURE: Vector {doc_id} was NOT found in the index after upsert.")