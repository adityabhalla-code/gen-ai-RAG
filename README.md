# gen-ai-RAG
RAG implementation
# RAG Steps
> 1. created document chunks
> 2. embedded document chunk and stored in a vector db
> 3. use similarity search or MMR to find the most relevant document to query
> 4. used zypher/7b LLM to query and summarize the retrieval using langchain
> 5. gradio implementation

## usage
> 1. Add your documents to data folder.
> 2. To prepare data for RAG as mentioned in setps 1-2 above, run below command
```python
python process_data.py
```
> 3. step 1 and 2 above will download the required model for embeddings. 
> 4. convert the text data to chunks and store in docs/chroma as a vector db. 
> 4. Run below command to launch the gradio app.
```python
python main.py
```
