import asyncio
import re
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rank_bm25 import BM25Okapi

# Retrieve relevant info from transcript using hybrid RAG with async improvements
async def retrieve_chunks(text, query, k=3):
    start_time = time.time()  # Start time for retrieval process
    text = text.replace('\r', '').replace('\xa0', ' ')
    text = re.sub(r'\n{2,}', '\n\n', text.strip())
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    vectordb = Chroma.from_texts(texts, embedding_model, metadatas=metadata, persist_directory="chroma_db")

    # Using BM25 and embedding models in parallel
    tokenized_texts = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # Perform the retrievals in parallel
    async def get_results():
        results_embedding = await asyncio.to_thread(vectordb.similarity_search_with_score, query, k)  # Use thread for DB search
        results_bm25 = [(i, bm25.get_scores(query.split())[i]) for i in range(len(texts))]
        return results_embedding, results_bm25

    results_embedding, results_bm25 = await get_results()

    results_bm25 = sorted(results_bm25, key=lambda x: x[1], reverse=True)[:k]
    results_bm25_docs = [(Document(page_content=texts[i], metadata=metadata[i]), score) for i, score in results_bm25]

    doc_lookup = {doc.page_content: doc for doc, _ in results_bm25_docs + results_embedding}

    def rrf(bm25_res, emb_res):
        scores = {}
        for r, (doc, _) in enumerate(bm25_res + emb_res):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (r + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = rrf(results_bm25_docs, results_embedding)
    end_time = time.time()  # End time for retrieval process
    print(f"Time taken for retrieval: {end_time - start_time:.4f} seconds")

    return [doc_lookup[doc_id] for doc_id, _ in fused[:k]]