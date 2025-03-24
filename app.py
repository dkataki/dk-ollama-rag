import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)  # Initialize PDF reader
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Extract text from each page
    return text

# Function to create FAISS vector store from PDF text
def faiss_store(text, path="faiss_index"):
    # Split text into chunks for better embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Use HuggingFace embedding model for text vectorization
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index from text chunks
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Save the FAISS index locally for later use
    vector_store.save_local(path)

# Function to load FAISS index if it exists
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(path):
        # Load existing FAISS index
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None  # Return None if the index is missing

# Function to build the RetrievalQA chain for querying
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)

    # If the index is missing, prompt the user to upload a PDF
    if not vector_store:
        st.error("FAISS index not found. Please upload a PDF to create one.")
        return None

    retriever = vector_store.as_retriever()  # Use FAISS index as retriever

    # Initialize LLaMA model from Ollama
    llm = Ollama(model="llama3.2")

    # Create a QA chain with the retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Streamlit UI setup
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

# File uploader widget to receive PDF from user
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Handle PDF upload and FAISS index creation
if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"

    # Ensure the upload directory exists
    os.makedirs("uploaded", exist_ok=True)

    # Save the uploaded PDF locally
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    if text.strip():
        st.info("Creating FAISS vector store...")
        faiss_store(text)  # Build and save the FAISS index

        st.info("Initializing chatbot...")
        st.session_state.qa_chain = build_qa_chain()  # Store chatbot in session

        st.success("Chatbot is ready!")
    else:
        st.warning("Failed to extract text. Please upload a valid PDF.")

# Ensure QA chain is available for querying
if 'qa_chain' in st.session_state and st.session_state.qa_chain:
    question = st.text_input("Ask a question about the uploaded PDF:")

    if question:
        st.info("Querying the document...")
        try:
            # Run the query through the chatbot using invoke() to handle multiple outputs
            response = st.session_state.qa_chain.invoke({"query": question})

            # Extract and display the main answer
            st.success(f"Answer: {response['result']}")

            # Optionally display source documents
            with st.expander("Source Documents"):
                for doc in response['source_documents']:
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"Error during query: {e}")
else:
    st.info("Please upload a PDF to get started.")