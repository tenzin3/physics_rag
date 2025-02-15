import json
from pathlib import Path
from typing import Dict, List
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def load_extracted_text(json_file: Path|str) -> Dict[int, str]:
    with open(json_file, "r") as f:
        return json.load(f)

def chunk_text_with_spacy(text: str, max_length: int = 500) -> List[str]:
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

def chunk_extracted_text(extracted_text: Dict[int, str], max_length: int = 500) -> Dict[int, List[str]]:
    chunked_text = {}
    for page_no, text in extracted_text.items():
        chunked_text[page_no] = chunk_text_with_spacy(text, max_length)
    return chunked_text

# Load extracted text from JSON file
extracted_text = load_extracted_text("extracted_text.json")

# Chunk the extracted text using SpaCy
chunked_text = chunk_extracted_text(extracted_text)

# Write the chunked text to a new JSON file
with open("chunked_text.json", "w") as f:
    json.dump(chunked_text, f, ensure_ascii=False, indent=2)
