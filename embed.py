from transformers import AutoTokenizer, AutoModel
import torch

class Embedder:
    def __init__(self, model_name="thellert/physbert_cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def token_embed(self, text:str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        token_embeddings = token_embeddings[:, 1:-1, :]
        return token_embeddings
    
    def embed(self, text:str):
        token_embeddings = self.token_embed(text)
        sentence_embedding = token_embeddings.mean(dim=1)
        return sentence_embedding.detach().numpy()  # Convert to NumPy array


def get_text_embedding(text:str):
    embedder = Embedder()
    return embedder.embed(text)

if __name__ == "__main__":
    from utils import read_json, write_json

    chunked_text = read_json("data/chunked_text.json")

    embeded_chunks = []

    embedder = Embedder()
    for page_no, chunks in chunked_text.items():
        for i, chunk in enumerate(chunks, start=1):
            embedding = embedder.embed(chunk)
            embeded_chunks.append({
                "page_no": page_no,
                "chunk_no": i,
                "text": chunk,
                "embedding": embedding.tolist()
            })
    write_json("data/embeded_chunks.json", embeded_chunks)