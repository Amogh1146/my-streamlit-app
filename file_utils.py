# utils/file_utils.py

import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text


def extract_text_from_docx(file_path):
    """
    Extract text from a DOCX file using python-docx.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text
