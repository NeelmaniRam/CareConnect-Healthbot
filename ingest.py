import os

# Ensure only one OpenMP runtime is used
os.environ["KMP_INIT_AT_FORK"] = "FALSE"  # Prevent issues with subprocesses
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow execution but risks remain

import torch
import faiss
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Limit PyTorch threads
torch.set_num_threads(1)

DATA_PATH = 'data/'  # Directory containing PDFs
VECTORSTORE_DIR = 'vectorstore/'  # Directory to save FAISS index and metadata

# Create vector database using FAISS
def create_vector_db():
    # Step 1: Load documents
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Step 3: Set up embeddings model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})

    # Step 4: Compute embeddings
    texts_content = [doc.page_content for doc in texts]
    embeddings_matrix = embeddings.embed_documents(texts_content)

    # Step 5: Initialize FAISS index
    embedding_dim = len(embeddings_matrix[0])  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is used for similarity search

    # Add embeddings to FAISS index
    index.add(embeddings_matrix)

    # Step 6: Save FAISS index and metadata
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))

    # Save metadata for future retrieval (text data and mappings)
    with open(os.path.join(VECTORSTORE_DIR, "vector_store.pkl"), "wb") as f:
        pickle.dump({"texts": texts_content}, f)

    print(f"FAISS index created and stored in '{VECTORSTORE_DIR}' successfully.")

if __name__ == "__main__":
    create_vector_db()
