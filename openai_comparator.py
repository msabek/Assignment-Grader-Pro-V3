from openai import OpenAI, APITimeoutError
import json
from typing import Dict, Any, List
import time
from datetime import datetime, timedelta

class OpenAIComparator:
    def __init__(self):
        """Initialize the OpenAIComparator."""
        self._api_key = None
        self._client = None
        self._cached_models = None
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)  # Cache models for 1 hour
        # Default models as fallback
        self._available_models = [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        
        # Define the grading function schema
        self.grading_function = {
            "type": "function",
            "function": {
                "name": "grade_assignment",
                "description": "Grade a student's assignment submission based on the provided solution and marking criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question_number": {"type": "integer"},
                                    "score": {"type": "number"},
                                    "max_score": {"type": "number"},
                                    "feedback": {"type": "string"}
                                },
                                "required": ["question_number", "score", "max_score", "feedback"]
                            }
                        },
                        "total_score": {"type": "number"},
                        "total_possible": {"type": "number", "enum": [20]},
                        "overall_feedback": {"type": "string"}
                    },
                    "required": ["questions", "total_score", "total_possible", "overall_feedback"]
                }
            }
        }

    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        if self._is_cache_valid():
            return self._cached_models or self._available_models
        return self._available_models

    def _is_cache_valid(self) -> bool:
        """Check if the cached models are still valid."""
        if not self._cache_timestamp or not self._cached_models:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration

    def _filter_models(self, models_data: List[Any]) -> List[str]:
        """Filter and validate OpenAI models."""
        filtered_models = set()  # Use set to avoid duplicates
        
        for model in models_data:
            try:
                model_id = getattr(model, 'id', None)
                if not isinstance(model_id, str) or not model_id.startswith('gpt-'):
                    continue
                
                # Only include chat models
                if 'gpt-' in model_id:
                    filtered_models.add(model_id)
            except Exception as e:
                print(f"Error processing model {model}: {str(e)}")
                continue
        
        # Convert to list and sort by version
        models_list = list(filtered_models)
        
        def sort_key(model_name):
            # Sort key function that prioritizes GPT-4 over GPT-3.5
            # and puts 'turbo' versions at the top
            if 'gpt-4' in model_name:
                if 'turbo' in model_name:
                    return (0, model_name)  # GPT-4 turbo first
                return (1, model_name)  # Other GPT-4 second
            return (2, model_name)  # GPT-3.5 last
        
        models_list.sort(key=sort_key)
        return models_list if models_list else self._available_models

    def set_api_key(self, api_key: str) -> List[str]:
        """Set the OpenAI API key, initialize client and fetch available models."""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Please enter a valid OpenAI API key")
        
        # Check if we can use cached models
        if (self._api_key == api_key and self._is_cache_valid() and 
            self._cached_models is not None):
            return self._cached_models
        
        try:
            # Initialize client with reasonable timeouts
            self._api_key = api_key
            self._client = OpenAI(api_key=api_key, timeout=30.0)
            
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Test API key validity with a models list request
                    response = self._client.models.list()
                    
                    if not response or not hasattr(response, 'data'):
                        raise ValueError("Invalid response from OpenAI API")
                    
                    # Filter and process models
                    filtered_models = self._filter_models(response.data)
                    
                    if not filtered_models:
                        raise ValueError("No compatible GPT models found")
                    
                    # Update cache
                    self._cached_models = filtered_models
                    self._cache_timestamp = datetime.now()
                    
                    return filtered_models
                    
                except APITimeoutError:
                    last_error = "Request timed out"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    break
            
            # If we get here, all retries failed
            error_msg = str(last_error).lower()
            if "timeout" in error_msg:
                raise ValueError("OpenAI API is not responding. Please try again in a few moments.")
            elif "invalid" in error_msg or "key" in error_msg or "auth" in error_msg:
                raise ValueError("Invalid API key. Please check your OpenAI API key and try again.")
            elif "rate" in error_msg:
                raise ValueError("Too many requests. Please wait a moment before trying again.")
            elif "insufficient_quota" in error_msg:
                raise ValueError("Your OpenAI API quota has been exceeded. Please check your usage limits.")
            else:
                raise ValueError(f"Unable to connect to OpenAI: {last_error}")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                raise ValueError("OpenAI API is not responding. Please try again in a few moments.")
            elif "unauthorized" in error_msg or "authentication" in error_msg:
                raise ValueError("Invalid API key. Please check your OpenAI API key and try again.")
            elif "rate" in error_msg:
                raise ValueError("Too many requests. Please wait a moment before trying again.")
            else:
                raise ValueError(f"Error connecting to OpenAI: {str(e)}")

    def compare_texts(self, student_text: str, solution_text: str, model_name: str = "gpt-4", 
                     marking_criteria: str = None, temperature: float = 0.1, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using OpenAI."""
        if not self._client:
            return {
                "detailed_feedback": "Error: OpenAI API key not set. Please provide your API key in the settings."
            }

        # Validate model name
        if not isinstance(model_name, str):
            model_name = self._available_models[0]
            print(f"Invalid model name type, falling back to {model_name}")
        elif model_name not in self.available_models:
            closest_match = min(self.available_models, key=lambda x: len(set(x) ^ set(model_name)))
            model_name = closest_match
            print(f"Model '{model_name}' not found, using closest match: {closest_match}")

        try:
            criteria_section = ""
            if marking_criteria and marking_criteria.strip():
                criteria_section = f"Use these marking criteria: {marking_criteria}"

            prompt = f'''Grade the student submission by comparing it with the solution. Provide structured feedback with scores and detailed comments for each question.

Solution:
{solution_text}

Student submission:
{student_text}

{criteria_section}'''

            max_retries = 3
            retry_delay = 1
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Get response from OpenAI with function calling
                    chat_completion = self._client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise grading assistant that provides structured feedback using function calls."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        functions=[self.grading_function["function"]],
                        function_call={"name": "grade_assignment"},
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=60.0  # 60 second timeout for completion requests
                    )
                    
                    # Process response
                    response = chat_completion.choices[0].message
                    
                    if response.function_call and response.function_call.arguments:
                        try:
                            # Parse the function call arguments
                            grading_result = json.loads(response.function_call.arguments)
                            
                            # Format the response
                            feedback_parts = []
                            for question in grading_result['questions']:
                                feedback_parts.append(
                                    f"Question {question['question_number']}: [Score: {question['score']}/{question['max_score']}] {question['feedback']}"
                                )
                            
                            feedback_parts.append(f"\nTotal Score: {grading_result['total_score']}/20")
                            if grading_result.get('overall_feedback'):
                                feedback_parts.append(f"\nOverall Feedback: {grading_result['overall_feedback']}")
                            
                            return {'detailed_feedback': '\n'.join(feedback_parts)}
                            
                        except json.JSONDecodeError:
                            raise ValueError("Invalid response format from OpenAI")
                    else:
                        # Fallback to content if function call is not present
                        return {'detailed_feedback': response.content or "No feedback provided"}
                        
                except APITimeoutError:
                    last_error = "Request timed out"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    break
            
            # If we get here, all retries failed
            error_msg = str(last_error).lower()
            if "timeout" in error_msg:
                return {
                    "detailed_feedback": "Error: The request took too long to process. Please try again."
                }
            elif "rate" in error_msg:
                return {
                    "detailed_feedback": "Error: Too many requests. Please wait a moment before trying again."
                }
            elif "quota" in error_msg:
                return {
                    "detailed_feedback": "Error: API quota exceeded. Please check your OpenAI account."
                }
            return {
                "detailed_feedback": f"Error processing submission: {last_error}"
            }
                
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                return {
                    "detailed_feedback": "Error: The request took too long to process. Please try again."
                }
            elif "invalid" in error_msg and "model" in error_msg:
                return {
                    "detailed_feedback": f"Error: The model '{model_name}' is not available. Please select a different model."
                }
            return {
                "detailed_feedback": f"Error: {str(e)}"
            }