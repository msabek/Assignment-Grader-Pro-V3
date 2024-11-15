import ollama
import json
from typing import Dict, Any, List, Optional, Sequence
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

class OllamaTextComparator:
    def __init__(self):
        """Initialize the OllamaTextComparator."""
        self._cached_models = None
        self._ollama_available = None
        self.available_models = self.get_available_models()
    
    def _check_ollama_connection(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception as e:
            if "connection refused" in str(e).lower():
                print("Ollama service not running - please start Ollama first")
            else:
                print(f"Ollama error: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        if self._cached_models is not None:
            return self._cached_models
        
        if not self._check_ollama_connection():
            return ["mistral", "llama2", "mixtral"]  # Default models
            
        try:
            response = ollama.list()
            if isinstance(response, dict) and 'models' in response:
                models = [model['name'] for model in response['models']]
                return sorted(models) if models else ["mistral", "llama2", "mixtral"]
            return ["mistral", "llama2", "mixtral"]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return ["mistral", "llama2", "mixtral"]

    def compare_texts(self, student_text: str, solution_text: str, model_name: str = "mistral", 
                     marking_criteria: Optional[str] = None, temperature: float = 0.1, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using Ollama."""
        # First check Ollama connection
        if not self._check_ollama_connection():
            return {
                "detailed_feedback": """Error: Ollama service is not running. Please install and start Ollama first:
1. Install Ollama: https://ollama.ai
2. Start Ollama service
3. Try again"""
            }
            
        try:
            if not isinstance(model_name, str) or model_name not in self.available_models:
                print(f"Invalid model name '{model_name}', falling back to {self.available_models[0]}")
                model_name = self.available_models[0]
            
            criteria_section = ""
            if marking_criteria and marking_criteria.strip():
                criteria_section = f"Use these marking criteria: {marking_criteria}"

            prompt = f'''Grade the student submission by comparing it with the solution. Return a JSON object with scores and feedback for each question and a total score. Format your response EXACTLY as:

{{"feedback": "Question 1: [Score: X/Y] <feedback>
Question 2: [Score: X/Y] <feedback>
Question 3: [Score: X/Y] <feedback>
Total Score: X/20"}}

Solution:
{solution_text}

Student submission:
{student_text}

{criteria_section}'''

            # Use ollama.chat for better response formatting with configurable parameters
            messages: Sequence[Message] = [
                Message(
                    role="system",
                    content="You are a precise grading assistant. Always respond with valid JSON objects only."
                ),
                Message(
                    role="user",
                    content=prompt
                )
            ]
            
            response = ollama.chat(
                model=model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            # Extract response content
            response_text = response['message']['content']
            
            try:
                # Parse response
                response_data = json.loads(response_text)
                
                if isinstance(response_data, dict):
                    if 'feedback' in response_data:
                        return {'detailed_feedback': response_data['feedback']}
                    
                    parts = []
                    for key, value in response_data.items():
                        if isinstance(value, dict):
                            score = value.get('score', value.get('gained_mark', value.get('marks', '')))
                            feedback = value.get('feedback', '')
                            parts.append(f"{key}:")
                            if score:
                                parts.append(f"Score: {score}")
                            if feedback:
                                parts.append(feedback)
                        else:
                            parts.append(f"{key}: {value}")
                    
                    return {'detailed_feedback': '\n'.join(parts)}
                    
            except json.JSONDecodeError:
                return {'detailed_feedback': response_text}
                
        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower():
                return {
                    "detailed_feedback": """Error: Unable to connect to Ollama service. Please ensure Ollama is installed and running on your system:
1. Install Ollama: https://ollama.ai
2. Start Ollama service
3. Try again"""
                }
            return {
                "detailed_feedback": f"Error processing submission: {type(e).__name__} - {error_msg}"
            }
