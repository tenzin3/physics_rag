from sklearn.metrics.pairwise import cosine_similarity
from embed import get_text_embedding
import numpy as np
from utils import read_json

def get_chunks_data():
    return read_json("data/embeded_chunks.json")

def retrieve_topk_chunks(query_text:str , top_k=5):
    # Generate embedding for the query text
    query_embedding = get_text_embedding(query_text)

    similarities = []
    chunks_data = get_chunks_data()
    for chunk in chunks_data:
        chunk_embedding = np.array(chunk['embedding'])
        sim = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]
        similarities.append((sim, chunk))

    # Sort by similarity in descending order
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

    # Return top-k chunks
    return similarities[:top_k]

if __name__ == "__main__":
    query = "Explain the role of blockchain in log analysis."
    top_chunks = retrieve_topk_chunks(query, top_k=3)

    for score, chunk in top_chunks:
        print(f"Score: {score:.4f}, Page: {chunk['page_no']}, Chunk: {chunk['chunk_no']}")
        print(f"Text: {chunk['text']}\n")
    
    pass 