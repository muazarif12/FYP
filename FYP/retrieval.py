import asyncio
import re
import time
import numpy as np
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import functools
import concurrent.futures

# Cache for embedding model to avoid reloading
_embedding_model = None

def get_embedding_model():
    """Singleton pattern to reuse the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        # Use device="cuda" only if you have CUDA available, otherwise use "cpu"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True}  # Batch processing for speed
        )
    return _embedding_model

# Precomputed Chroma index (to be initialized on first call)
_chroma_db = None
_index_texts = None
_index_metadata = None
_collection_name = str(uuid.uuid4())  # Use a unique collection name for each session

async def precompute_chroma_index(text):
    """Precompute and cache the Chroma index for faster subsequent retrievals"""
    global _chroma_db, _index_texts, _index_metadata
    
    if _chroma_db is not None:
        return _chroma_db, _index_texts, _index_metadata
    
    # Preprocess text
    text = text.replace('\r', '').replace('\xa0', ' ')
    text = re.sub(r'\n{2,}', '\n\n', text.strip())
    docs = [Document(page_content=text)]
    
    # Use larger chunk size with smaller overlap for efficiency
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    _index_texts = [chunk.page_content for chunk in chunks]
    _index_metadata = [chunk.metadata for chunk in chunks]
    
    # Get embedding model
    embedding_model = get_embedding_model()
    
    # Create Chroma index
    # Using in-memory Chroma DB with GPU acceleration
    # persist_directory=None means in-memory
    _chroma_db = await asyncio.to_thread(
        Chroma.from_texts,
        _index_texts,
        embedding_model,
        _index_metadata,
        collection_name=_collection_name
    )
    
    return _chroma_db, _index_texts, _index_metadata

async def retrieve_chunks(text, query, k=3):
    """Enhanced retrieval using Chroma and optimized for performance"""
    start_time = time.time()
    
    # Get or create Chroma index
    chroma_db, texts, metadata = await precompute_chroma_index(text)
    
    # Create BM25 index if not already cached with the index
    if not hasattr(chroma_db, '_bm25_index'):
        tokenized_texts = [t.split() for t in texts]
        chroma_db._bm25_index = BM25Okapi(tokenized_texts)
    
    bm25 = chroma_db._bm25_index
    
    # Run embedding and BM25 searches in parallel
    async def run_parallel_searches():
        # Define synchronous functions for thread pool
        def chroma_search():
            return chroma_db.similarity_search_with_score(query, k=k)
        
        def bm25_search():
            scores = bm25.get_scores(query.split())
            indices = np.argsort(scores)[-k:][::-1]  # Get top k indices
            return [(i, scores[i]) for i in indices]
        
        # Execute in thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks
            chroma_future = executor.submit(chroma_search)
            bm25_future = executor.submit(bm25_search)
            
            # Get results
            results_embedding = chroma_future.result()
            results_bm25_raw = bm25_future.result()
        
        # Convert BM25 results to documents
        results_bm25 = [(Document(page_content=texts[i], metadata=metadata[i] if i < len(metadata) else {}), score) 
                       for i, score in results_bm25_raw]
        
        return results_embedding, results_bm25
    
    # Get search results
    results_embedding, results_bm25 = await run_parallel_searches()
    
    # Create lookup table for documents
    doc_lookup = {doc.page_content: doc for doc, _ in results_bm25 + results_embedding}
    
    # Reciprocal Rank Fusion for better results
    def rrf(bm25_res, emb_res, k1=60):
        """Improved RRF with k parameter to control influence"""
        scores = {}
        # Process BM25 results
        for r, (doc, _) in enumerate(bm25_res):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (r + k1)
        
        # Process embedding results
        for r, (doc, _) in enumerate(emb_res):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (r + k1)
            
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Apply fusion ranking
    fused = rrf(results_bm25, results_embedding)
    
    end_time = time.time()
    print(f"Time taken for retrieval: {end_time - start_time:.4f} seconds")
    
    # Return top k documents
    return [doc_lookup[doc_id] for doc_id, _ in fused[:k]]

# Function to initialize indexes in the background (call at application startup)
async def initialize_indexes(text):
    """Initialize indexes in background to speed up first query"""
    print("Pre-computing embedding indexes...")
    await precompute_chroma_index(text)
    print("Embedding indexes ready.")