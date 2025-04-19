A Retrieval-Augmented Generation (RAG) pipeline consists of two key components:
1. Retriever: Searches a knowledge base for relevant documents based on the user’s query.
2. Generator: Uses retrieved documents as context to generate accurate and relevant responses.

RAG improves LLMs by reducing hallucinations through real-world context, ensuring responses are more accurate and grounded in factual information. It also keeps answers up-to-date by retrieving the latest knowledge, eliminating the need for frequent retraining. Additionally, by incorporating external data sources, RAG significantly enhances the factual accuracy of AI-generated responses, which makes LLMs more reliable and context-aware

Building a RAG Pipeline for LLMs:
In our implementation, we will:
1. Use Wikipedia as our external knowledge source.
2. Employ Sentence Transformers for embedding text and FAISS for efficient similarity search.
3. Utilize Hugging Face’s question-answering pipeline to extract answers from retrieved documents.
