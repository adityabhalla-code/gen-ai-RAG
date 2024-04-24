from langchain_community.llms import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from getpass import getpass
import os

# # Enter your HuggingFace access token when prompted
pass_token = getpass("Enter your HuggingFace access token: ")
os.environ["HF_TOKEN"] = pass_token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = pass_token
del pass_token

# Load the all-MiniLM-L6-v2 model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

class MyEmbeddingClass:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        embedding_array = self.model.encode(query)  # Generate the embedding
        embedding_list = embedding_array.tolist()  # Convert numpy array to list
        return embedding_list

# class instance
embedding_function = MyEmbeddingClass(model)


# vector db - load from disk
persist_directory = 'docs/chroma/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
vectordb.get()
print(f"Total items in vector db--{vectordb._collection.count()}")


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens = 512,
    top_k = 30,
    temperature = 0.1,
    repetition_penalty = 1.03,
)


# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
)

question = "Requirements for a canada visa?"

import pandas as pd
def documents_to_dataframe(documents):
    data = [{'page': doc.metadata['page'], 'source': doc.metadata['source']} for doc in documents]
    df = pd.DataFrame(data)
    return df

# Invoke the chain
try:
    result = qa_chain.invoke({"query": question})
    print(type(result))
    print(result.keys())
    print(result["result"])
    print(type(result['source_documents']))
    print(result['query'])
    df = documents_to_dataframe(result['source_documents'])
    print(df.head(2))
except ValueError as e:
    print("ValueError:", e)
