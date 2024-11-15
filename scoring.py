from scoring_criteria import ScoringCriteria
import re

class ScoreCalculator:
    def __init__(self):
        self.criteria = ScoringCriteria()
    
    def update_scoring_criteria(self, new_criteria):
        """Update the scoring criteria."""
        self.criteria.update_criteria(new_criteria)
    
    def reset_scoring_criteria(self):
        """Reset scoring criteria to default."""
        self.criteria.reset_to_default()
    
    def extract_score(self, detailed_feedback):
        """Extract score from detailed feedback."""
        try:
            # Look for various score patterns
            patterns = [
                r'(?:Total Score|Total|Final Score):\s*(\d+(?:\.\d+)?)/20',
                r'total_gained_mark[s]?["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'gained_mark[s]?["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*/\s*20'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, str(detailed_feedback), re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    return round(max(0, min(20, score)), 1)
            
            # If no explicit total score found, try to sum up individual question scores
            question_scores = re.findall(r'(?:score|gained_mark)[s]?["\']?\s*:\s*(\d+(?:\.\d+)?)', str(detailed_feedback), re.IGNORECASE)
            if question_scores:
                total_score = sum(float(score) for score in question_scores)
                return round(max(0, min(20, total_score)), 1)
            
            return 0.0
                
        except Exception as e:
            print(f"Error extracting score: {str(e)}")
            return 0.0
    
    def calculate_score(self, detailed_feedback):
        """Calculate score from detailed feedback (legacy method)."""
        return self.extract_score(detailed_feedback)
    
    def generate_feedback(self, detailed_feedback, final_score):
        """Generate detailed feedback based on comparison results and current criteria."""
        criteria = self.criteria.get_criteria()
        feedback = []
        
        # Use the AI-generated detailed feedback
        feedback.append(detailed_feedback)
        
        # Add score-based feedback
        if final_score >= 18:
            feedback.append("\nOutstanding performance!")
        elif final_score >= 15:
            feedback.append("\nVery good understanding shown.")
        elif final_score >= 10:
            feedback.append("\nSatisfactory work, but review the material.")
        else:
            feedback.append("\nSignificant revision needed. Please review the material.")
        
        return "\n".join(feedback)
