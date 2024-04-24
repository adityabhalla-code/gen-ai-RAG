"""
Query the vector db using similarity search
"""
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

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


def display_results(returned_documents):
    for doc in returned_documents:
        print("Metadata of the returned documents:")
        print(doc.metadata)
        text = doc.page_content
        clean_text = ' '.join(text.split())
        print(f"Answer: {clean_text}")


question = "Requirements for a canada visa?"

print(f'{"--" *100}Similarity search{"--" *100}')
returned_docs = vectordb.similarity_search(question,k=10) # k --> No. of doc as return
print(f"Total documents returned using similarity search --- {len(returned_docs)}")
display_results(returned_docs)

# similarity search with MMR
print(f'{"--" *100}Similarity search MMR{"--" *100}')
docs_with_mmr=vectordb.max_marginal_relevance_search(question, k=3, fetch_k=6) # With MMR
print(f"Total documents returned using MMR --- {len(returned_docs)}")
display_results(docs_with_mmr)
