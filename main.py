from langchain_community.llms import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from getpass import getpass
import gradio as gr
import pandas as pd
import os

# Load the all-MiniLM-L6-v2 model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Enter your HuggingFace access token when prompted
pass_token = getpass("Enter your HuggingFace access token: ")
os.environ["HF_TOKEN"] = pass_token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = pass_token
del pass_token
# Load LLM as HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03,
)

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

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
)

def get_docs_metadata(documents):
    data = [{'page': doc.metadata['page'], 'source': doc.metadata['source']} for doc in documents]
    df = pd.DataFrame(data)
    return df

def get_answer(question):
    result = qa_chain.invoke({"query": question})
    source_documents = result['source_documents']
    df = get_docs_metadata(source_documents)
    df.to_csv("df.csv", index=False)
    return result['result'], df





def run_app():
    with gr.Blocks() as app:
        gr.Markdown("## RAG Implementation")
        gr.Markdown("Enter a question related to the document and click submit.")
        output_source_docs = gr.DataFrame(label="Source Documents")
        output_answer = gr.Textbox(label="Answer", interactive=False, placeholder="Answers will appear here...")
        with gr.Row():
            with gr.Column(scale=6):
                input_question = gr.Textbox(lines=2, placeholder="Type your document-related question here...",
                                            label="Question")
            with gr.Column(scale=6):
                submit_button = gr.Button("Submit")
        def on_button_click(question):
            answer, source_docs = get_answer(question)
            return answer, source_docs
        submit_button.click(on_button_click, inputs=input_question, outputs=[output_answer, output_source_docs])
    app.launch()


if __name__ == "__main__":
    run_app()

