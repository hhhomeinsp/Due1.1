import io
import docx
import PyPDF2
import pandas as pd

def extract_text_from_file(file):
    try:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            return file.getvalue().decode("utf-8")
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file.getvalue()))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            return "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(file.getvalue()))
            return df.to_string(index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise ValueError(f"Error extracting text from file: {str(e)}")
