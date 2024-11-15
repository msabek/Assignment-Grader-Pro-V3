import PyPDF2
import re

class PDFProcessor:
    def __init__(self):
        pass

    def extract_text(self, pdf_file):
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def process_text(self, text):
        """Process extracted text using basic text processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                     'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                     'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        words = text.split()
        processed_words = [word for word in words if word not in stop_words]
        
        return ' '.join(processed_words)
