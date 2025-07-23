import logging
import docx2txt
import PyPDF2

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logger.debug(f"Extracted text from PDF {file_path}: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX {file_path}: {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.debug(f"Extracted text from TXT {file_path}: {text[:200]}...")
            return text
    except Exception as e:
        logger.error(f"Error extracting TXT {file_path}: {str(e)}")
        return ""

def extract_text(file_path):
    logger.debug(f"Processing file: {file_path}")
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        return ""