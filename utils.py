import fitz
import json 
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



pdf_file = Path("research_paper.pdf")
extracted_text = extract_text_from_pdf_file(pdf_file)
write_json("extracted_text.json", extracted_text)