from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextComparator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def compare_texts(self, student_text, solution_text):
        """Compare student text with solution text using TF-IDF and cosine similarity."""
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([student_text, solution_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        except Exception as e:
            raise Exception(f"Error comparing texts: {str(e)}")

    def find_missing_concepts(self, student_text, solution_text):
        """Identify key concepts present in solution but missing in student text."""
        solution_words = set(solution_text.split())
        student_words = set(student_text.split())
        
        missing_concepts = solution_words - student_words
        return list(missing_concepts)
