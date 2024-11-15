import requests
import json
import re
from typing import Dict, Any, Optional

class OpenAICompatibleComparator:
    def __init__(self):
        self._api_base = "http://127.0.0.1:1234/v1"
        self._available = None
        self._cached_models = None
        # Default models that should work with most OpenAI-compatible servers
        self.available_models = ["mistral-7b-instruct-v0.3", "llama2", "mixtral", "qwen2-0.5b-instruct"]

    def set_api_base(self, api_base: str) -> None:
        # Normalize URL
        api_base = api_base.strip().rstrip('/')
        if not api_base.startswith(('http://', 'https://')):
            api_base = f"http://{api_base}"
        if not api_base.endswith('/v1'):
            api_base = f"{api_base}/v1"
        self._api_base = api_base

    def get_connection_status(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self._api_base}/models", timeout=5)
            if response.status_code == 200:
                self._available = True
                models_data = response.json()
                available_models = []
                if isinstance(models_data, dict) and 'data' in models_data:
                    available_models = [model['id'] for model in models_data['data']]
                
                # If no models found from API, use default models
                if not available_models:
                    available_models = self.available_models
                else:
                    # Update available_models with API response
                    self.available_models = available_models
                    
                return {
                    'available': True,
                    'api_base': self._api_base,
                    'error': None,
                    'models': available_models
                }
        except Exception as e:
            self._available = False
            base_url = self._api_base.rsplit('/v1', 1)[0]
            error_msg = f'''Server connection failed:
1. Ensure LM Studio server is running (check Developer tab)
2. Verify port 1234 is accessible
3. Try these endpoints in your browser:
   - {base_url}/health
   - {base_url}/models
4. Check LM Studio server logs
5. Ensure no other app is using port 1234

Error details: {str(e)}'''
            return {
                'available': False,
                'api_base': self._api_base,
                'error': error_msg,
                'models': self.available_models  # Return default models when server is unavailable
            }

    def compare_texts(self, student_text: str, solution_text: str, 
                     model_name: str = "mistral-7b-instruct-v0.3", 
                     marking_criteria: Optional[str] = None,
                     temperature: float = 0.1,
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using local OpenAI-compatible server."""
        if not marking_criteria or not marking_criteria.strip():
            return {
                "detailed_feedback": "Error: Marking criteria is required"
            }
            
        try:
            # Check server connection
            try:
                response = requests.get(f"{self._api_base}/models", timeout=5)
                if response.status_code != 200:
                    return {
                        "detailed_feedback": "Error: Local OpenAI-compatible server is not running. Please ensure the server is running and accessible."
                    }
            except Exception as e:
                return {
                    "detailed_feedback": f"Error: Cannot connect to server at {self._api_base}. Please check if it's running."
                }

            # Prepare grading prompt with strict format
            prompt = f'''Grade this student submission according to EXACTLY these marking criteria and nothing else:

{marking_criteria}

Solution:
{solution_text}

Student submission:
{student_text}

Return your response in this format:
{{
    "feedback": "Question 1: [Score: X/Y] <feedback>
    Question 2: [Score: X/Y] <feedback>
    Question 3: [Score: X/Y] <feedback>
    Total Score: X/20"
}}'''

            # Make request to local server with 3-minute timeout
            response = requests.post(
                f"{self._api_base}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise grading assistant. Always follow the marking criteria exactly and provide scores in the specified format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False  # Make sure streaming is disabled
                },
                timeout=180  # 3-minute timeout for long-running evaluations
            )

            if response.status_code != 200:
                return {
                    "detailed_feedback": f"Error: Server returned status {response.status_code}. Please ensure the server is running and the model is loaded."
                }

            # Parse response
            result = response.json()
            if 'choices' not in result or not result['choices']:
                return {
                    "detailed_feedback": "Error: Invalid response format from server"
                }

            content = result['choices'][0]['message']['content']
            
            try:
                # Try to parse as JSON first
                response_data = json.loads(content)
                if isinstance(response_data, dict) and 'feedback' in response_data:
                    return {'detailed_feedback': response_data['feedback']}
                
                # If JSON parsing succeeds but format is different
                feedback_parts = []
                for key, value in response_data.items():
                    if isinstance(value, dict):
                        score = value.get('score', value.get('gained_mark', value.get('marks', '')))
                        feedback = value.get('feedback', '')
                        feedback_parts.append(f"{key}:")
                        if score:
                            feedback_parts.append(f"Score: {score}")
                        if feedback:
                            feedback_parts.append(feedback)
                    else:
                        feedback_parts.append(f"{key}: {value}")
                
                return {'detailed_feedback': '\n'.join(feedback_parts)}
                    
            except json.JSONDecodeError:
                # If not valid JSON, try to find the total score using regex
                score_pattern = r'(?:Total Score|Score):\s*(\d+(?:\.\d+)?)/20'
                score_match = re.search(score_pattern, content, re.IGNORECASE)
                
                if score_match:
                    return {"detailed_feedback": content}
                else:
                    # If no total score found, return the content as is
                    return {"detailed_feedback": content}
                    
        except Exception as e:
            return {
                "detailed_feedback": f"Error: {str(e)}"
            }
