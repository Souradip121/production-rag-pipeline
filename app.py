import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import RAG components
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from pinecone import Pinecone as PineconeClient

# --- LOAD ENV VARS (at the top) ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Custom RAG Bot", page_icon="🧩", layout="centered")


# --- CORE LOGIC ---
def load_and_embed_data(source, source_type, namespace):
    with st.spinner(f"Processing your {source_type}..."):
        # 1. Load documents
        try:
            if source_type == "URL":
                loader = WebBaseLoader(source)
            else: # Handle PDF and TXT file uploads
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{source_type.lower()}") as tmp_file:
                    tmp_file.write(source.getvalue())
                    file_path = tmp_file.name
                
                if source_type == "PDF":
                    loader = PyPDFLoader(file_path)
                elif source_type == "TXT":
                    loader = TextLoader(file_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

        # 2. Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # 3. Initialize Pinecone client and index
        index_name = "production-rag-pipeline"
        try:
            pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(index_name)
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            st.warning(f"Could not clear namespace (might be empty or new): {e}")

        # 4. Embed and upsert new documents
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        vectorstore = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=index_name, namespace=namespace
        )
        st.success("Data successfully embedded!")
        
        # 5. Create and return a retriever with explicit namespace configuration
        # THIS IS THE KEY FIX: Forcing the namespace in the search arguments
        retriever = vectorstore.as_retriever(
            search_kwargs={'namespace': namespace}
        )
        return retriever

@st.cache_resource
def get_llm():
    return AzureChatOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"), temperature=0.0)

# --- SIDEBAR UI ---
st.sidebar.title("📚 Knowledge Base")
st.sidebar.write("Upload a document, enter a URL, and click 'Load' to build a knowledge base for the bot.")
source_type = st.sidebar.selectbox("Choose source type", ["PDF", "URL", "TXT"])
namespace_id = "custom_rag_data" 

if source_type == "URL":
    source_input = st.sidebar.text_input("Enter URL")
else:
    source_input = st.sidebar.file_uploader(f"Upload {source_type}", type=[source_type.lower()])

if st.sidebar.button("Load Data"):
    if source_input:
        retriever = load_and_embed_data(source_input, source_type, namespace_id)
        if retriever:
            st.session_state.retriever = retriever
            st.sidebar.success("Data loaded! Ready to answer questions.")
    else:
        st.sidebar.error("Please provide a source.")

# --- MAIN CHAT INTERFACE ---
st.title("🧩 Custom RAG Query Bot")
st.write("Load a document from the sidebar, then ask questions about its content.")

if "retriever" in st.session_state and st.session_state.retriever is not None:
    llm = get_llm()
    rag_prompt = ChatPromptTemplate.from_template("Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {input}")
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
    
    user_question = st.text_input("Ask your question here:")
    if user_question:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_question})
            st.subheader("Answer:")
            st.write(response['answer'])
            with st.expander("Show Source Documents"):
                for doc in response['context']:
                    st.info(doc.page_content)
else:
    st.info("Please load a data source from the sidebar to begin.")