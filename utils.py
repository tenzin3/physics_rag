import fitz
import json 
import spacy 
from pathlib import Path
from typing import Dict 

def extract_text_from_pdf_file(pdf_file_path: Path|str) -> Dict[int, str]:
    """Reads the content of a PDF file using PyMuPDF."""
    text = {}

    try:
        pdf_document = fitz.open(pdf_file_path)
        for no in range(len(pdf_document)):
            page = pdf_document.load_page(no)
            text[no+1] = page.get_text() if page.get_text() else ""
        return text
    except Exception as e:
        print(f"Failed to read PDF file {pdf_file_path}: {e}")
        return None 
    
def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_entities(text: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.label_)

    # Return unique entities
    entities = list(set(entities))
    return entities


if __name__ == "__main__":
    
    extracted_text = read_json("data/extracted_text.json")
    text = ""
    for page_no, page_text in extracted_text.items():
        text += page_text

    entities = get_entities(text)
    Path("entities.txt").write_text("\n".join(entities))