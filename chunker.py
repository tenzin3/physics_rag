import json
from pathlib import Path
from typing import Dict, List
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

class Chunker:
    def chunk_text_with_spacy(self, text: str, max_length: int) -> List[str]:
        doc = nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in doc.sents:
            sentence = sent.text
            sentence_length = len(sentence)
            if current_length + sentence_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk(self, extracted_text: Dict[int, str], max_length: int = 500) -> Dict[int, List[str]]:
        chunked_text = {}
        for page_no, text in extracted_text.items():
            chunked_text[page_no] = self.chunk_text_with_spacy(text, max_length)
        return chunked_text

