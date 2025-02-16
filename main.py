from pathlib import Path 
from typing import Dict 

from chunker import Chunker
from embed import Embedder
from utils import extract_text_from_pdf_file, write_json


def remove_paper_references(extracted_text:Dict)->Dict:
    res = {}
    for page_no, text in extracted_text.items():
        if page_no <= 12:
            res[page_no] = text

    return res


def main(pdf_file: Path| str, output_path: Path | str):
    """
    1.extract text from the PDF file
    2.chunk the extracted text
    3.embed the chunks
    """
    # Extract text from the PDF file
    extracted_text = extract_text_from_pdf_file(pdf_file)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    extracted_text = remove_paper_references(extracted_text)
    write_json(output_path / "extracted_text.json", extracted_text)


    # Chunk the extracted text
    chunker = Chunker()
    chunked_text = chunker.chunk(extracted_text)
    write_json(output_path/ "chunked_text.json", chunked_text)

    # Embed the chunks
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
    write_json(output_path/ "embeded_chunks.json", embeded_chunks)


if __name__ == "__main__":
    pdf_file = "data/research_paper.pdf"
    output_path = "data"
    main(pdf_file, output_path)
    