import PyPDF2
import io
import zipfile
import xml.etree.ElementTree as ET
import re

class FileProcessor:
    def __init__(self):
        pass

    def extract_text(self, file):
        """Extract text from uploaded file based on its type."""
        try:
            # Get file extension from filename
            file_ext = file.name.lower().split('.')[-1]

            if file_ext == 'pdf':
                return self._extract_from_pdf(file)
            elif file_ext == 'txt':
                return self._extract_from_txt(file)
            elif file_ext == 'docx':
                return self._extract_from_docx(file)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def extract_student_id(self, filename: str) -> str:
        """Extract student ID from filename."""
        try:
            # Common patterns for student IDs in filenames
            patterns = [
                r'^(\d{6,10})[\s_-]',  # Numbers at start followed by separator
                r'(?:id|student|number)[\s_-]?(\d{6,10})',  # ID/student/number followed by numbers
                r'[\s_-](\d{6,10})\.pdf$'  # Numbers before .pdf extension
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # If no match found, extract first sequence of 6-10 digits
            digits_match = re.search(r'(\d{6,10})', filename)
            if digits_match:
                return digits_match.group(1)
            
            # If no valid student ID found
            return "unknown"
            
        except Exception as e:
            print(f"Error extracting student ID: {str(e)}")
            return "unknown"

    def _extract_from_pdf(self, pdf_file):
        """Extract text from PDF file."""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()

    def _extract_from_txt(self, txt_file):
        """Extract text from text file."""
        text = txt_file.read().decode('utf-8')
        return text.strip()

    def _extract_from_docx(self, docx_file):
        """Extract text from Word document using zipfile."""
        try:
            # Read the file into memory
            docx_bytes = io.BytesIO(docx_file.read())
            
            # Open as zip file
            doc = zipfile.ZipFile(docx_bytes)
            
            # Read the main document content
            xml_content = doc.read('word/document.xml')
            
            # Parse XML
            tree = ET.fromstring(xml_content)
            
            # Extract text from all paragraphs (removing XML namespace)
            ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            paragraphs = []
            
            for paragraph in tree.findall(f'.//{ns}p'):
                texts = []
                for node in paragraph.findall(f'.//{ns}t'):
                    if node.text:
                        texts.append(node.text)
                if texts:
                    paragraphs.append(''.join(texts))
            
            return '\n'.join(paragraphs)
            
        except Exception as e:
            raise Exception(f"Error processing Word document: {str(e)}")

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
