# dk-ollama-rag
Ollama Commands.tx - Cheatsheet of Ollama commands
RAG Chatbot_DK.doc - the whole flow and code
app.py - main code engine
load_libs.txt - PIP required libraries

## Core Concepts ##

**Embeddings:** Numerical representations of text that capture semantic meaning. We use Nebius AI's embedding API and, in many notebooks, also the BAAI/bge-en-icl embedding model.

**Vector Store:** A simple database to store and search embeddings. We create our own SimpleVectorStore class using NumPy for efficient similarity calculations.

**Cosine Similarity:** A measure of similarity between two vectors. Higher values indicate greater similarity.

**Chunking:** Dividing text into smaller, manageable pieces. We explore various chunking strategies.

**Retrieval:** The process of finding the most relevant text chunks for a given query.

**Generation:** Using a Large Language Model (LLM) to create a response based on the retrieved context and the user's query. We use the meta-llama/Llama-3.2-3B-Instruct model via Nebius AI's API.

**Evaluation:** Assessing the quality of the RAG system's responses, often by comparing them to a reference answer or using an LLM to score relevance.
