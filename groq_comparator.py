import os
from groq import Groq
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class GroqTextComparator:
    def __init__(self):
        """Initialize the GroqTextComparator with API client."""
        self._api_key = os.environ.get('GROQ_API_KEY')
        if not self._api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        try:
            self.client = Groq(api_key=self._api_key)
            self._cached_models = None
            self.available_models = self.get_available_models()
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client: {str(e)}")
        
    def get_available_models(self) -> List[str]:
        """Fetch and cache available models from Groq API."""
        if self._cached_models is not None:
            return self._cached_models

        # Known Groq model prefixes
        groq_model_prefixes = ['mixtral', 'llama']
        fallback_models = ["mixtral-8x7b-32768", "llama2-70b-4096"]
        
        try:
            # Make an API call to list models
            models = self.client.models.list()
            available_models = []
            
            # Filter and process the models
            if hasattr(models, 'data'):
                for model in models.data:
                    if (
                        hasattr(model, 'id') and 
                        isinstance(model.id, str) and 
                        any(prefix in model.id.lower() for prefix in groq_model_prefixes)
                    ):
                        available_models.append(model.id)
            
            # Sort models by name for consistent ordering
            available_models.sort()
            
            # If no valid models found, use fallback models
            if not available_models:
                print("No models found from API, using fallback models")
                available_models = fallback_models
            
            # Cache the results
            self._cached_models = available_models
            print(f"Available Groq models: {available_models}")
            return self._cached_models
            
        except Exception as e:
            print(f"Error fetching models from Groq API: {type(e).__name__} - {str(e)}")
            # Fallback to default models
            self._cached_models = fallback_models
            return self._cached_models

    def compare_texts(self, student_text: str, solution_text: str, model_name: str = "mixtral-8x7b-32768", marking_criteria: str = None, temperature: float = 0.1, max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using Groq LLM."""
        if not isinstance(model_name, str) or model_name not in self.available_models:
            print(f"Invalid model name '{model_name}', falling back to {self.available_models[0]}")
            model_name = self.available_models[0]
            
        try:
            criteria_section = ""
            if marking_criteria and marking_criteria.strip():
                criteria_section = f"""Use the following marking criteria:
                {marking_criteria}"""

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

            # Get response from Groq
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are a precise grading assistant. Always respond with valid JSON objects only."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract and parse response directly
            response_text = chat_completion.choices[0].message.content
            
            try:
                # Parse response directly without sanitization
                response = json.loads(response_text)
                
                # Extract feedback, handling different possible response structures
                if isinstance(response, dict):
                    # If we have a simple feedback field, use it directly
                    if 'feedback' in response:
                        return {'detailed_feedback': response['feedback']}
                    
                    # If we have a nested structure, try to convert it to text
                    parts = []
                    for key, value in response.items():
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
                # If not valid JSON, try to extract score information using regex
                score_pattern = r'(\d+)(?:/|\s*out\s*of\s*)20'
                feedback = response_text
                matches = re.findall(score_pattern, response_text)
                if matches:
                    total_score = matches[-1]  # Take the last number as total
                    feedback = f"{response_text}\nTotal Score: {total_score}/20"
                
                return {'detailed_feedback': feedback}
                
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return {
                "detailed_feedback": f"Error processing submission: {str(e)}"
            }
