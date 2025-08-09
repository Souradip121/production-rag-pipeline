import os
from dotenv import load_dotenv

# For the RAG pipeline
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
# Import the components for the modern RAG chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import textwrap

print("Starting RAG query process...")
load_dotenv()

# --- 1. Initialize models ---

print("Initializing embeddings and LLM...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# --- 2. Initialize the Pinecone Vector Store ---

print("Initializing Pinecone vector store...")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="production-rag-pipeline",
    embedding=embeddings,
    namespace="main-kb"
)
retriever = vectorstore.as_retriever()

# --- 3. Create the Modern RAG Chain ---

print("Creating modern RAG chain...")

# This is the prompt template that will be sent to the LLM
rag_prompt = ChatPromptTemplate.from_template(
    "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {input}"
)

# This chain takes the user's question and the retrieved documents and formats the prompt
document_chain = create_stuff_documents_chain(llm, rag_prompt)

# This is the full chain that combines the retriever and the document chain
rag_chain = create_retrieval_chain(retriever, document_chain)


# --- 4. Ask a Question ---
print("\n--- Ready to answer questions! ---")
question = "What were the first words spoken on the moon?"
print(f"Question: {question}")

# Invoke the modern chain. Note the input key is "input".
result = rag_chain.invoke({"input": question})

# The answer is now in the 'answer' key
print("\nAnswer:")
print(textwrap.fill(result['answer'], width=80))

# The retrieved documents are in the 'context' key
print("\nSource Documents:")
for doc in result['context']:
    print(textwrap.fill(f"- {doc.page_content}", width=80))