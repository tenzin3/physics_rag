from transformers import AutoTokenizer, AutoModel
import torch

class Embedder:
    def __init__(self, model_name="thellert/physbert_cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def token_embed(self, text:str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        token_embeddings = token_embeddings[:, 1:-1, :]
        return token_embeddings
    
    def embed(self, text:str):
        token_embeddings = self.token_embed(text)
        sentence_embedding = token_embeddings.mean(dim=1)
        return sentence_embedding.detach().numpy()  # Convert to NumPy array

if __name__ == "__main__":
    from utils import read_json, write_json

    chunked_text = read_json("data/chunked_text.json")