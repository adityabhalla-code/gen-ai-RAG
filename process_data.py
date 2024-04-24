from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma # Light-weight and in memory
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


# Load the all-MiniLM-L6-v2 model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
persist_directory = 'docs/chroma/'

import torch
import os

def create_document_loader(data_dir="data"):
    abs_path = os.path.abspath(data_dir)
    loaders = []
    for root, dirs, files in os.walk(abs_path):
        for file_name in files:
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(root, file_name)
                loader = PyPDFLoader(file_path)
                loaders.append(loader)
    return loaders

def create_chunks(docs):
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    return splits

# Function to generate embeddings for a list of texts
def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings


# Create document loaders
document_loaders = create_document_loader()
print(f"Total documents found in loader--{len(document_loaders)}")
docs = []
for loader in document_loaders:
    loaded_docs = loader.load()
    docs.extend(loaded_docs)
    print(f"Loaded {len(loaded_docs)} documents from {loader}")
print(f"Total documents loaded using pyPDF: {len(docs)}")
print(docs[0].page_content)
# Document chunking
text_chunks = create_chunks(docs)
print(f"Total chunks created--{len(text_chunks)}")
# Extract the text content from the document chunks
chunk_texts = [chunk.page_content for chunk in text_chunks]
print(f"Total chunk texts--{len(chunk_texts)}")
# Create Embeddings
# Generate embeddings for the document chunks
chunk_embeddings = generate_embeddings(chunk_texts)
# Print the embeddings
# for i, embedding in enumerate(chunk_embeddings):
#     print(f"Lenght of embedding for the chunk :{len(embedding)}")
    # print(f"Embedding for chunk {i+1}: {embedding}")
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Create the vector database
vectordb = Chroma.from_documents(
    documents=text_chunks,  # splits we created earlier
    embedding=embedding_function,
    persist_directory=persist_directory
    # persist_directory="./chroma_db" ,
)
vectordb.persist() # save vectordb
print(f"Total items in vector db--{vectordb._collection.count()}") # same as number of splites
