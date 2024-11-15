from mistralai import Mistral
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import time

class MistralComparator:
    def __init__(self):
        """Initialize the MistralComparator."""
        self._api_key = None
        self._client = None
        self._cached_models = None
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)  # Cache models for 1 hour
        # Default models as fallback - latest versions
        self._available_models = [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-tiny-latest"
        ]
        
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
        """Filter and validate Mistral models."""
        filtered_models = []
        
        # Known model prefixes and capabilities
        known_prefixes = ["mistral-", "mixtral-"]
        model_tiers = ["tiny", "small", "medium", "large"]
        
        for model in models_data:
            try:
                # Handle different response formats
                if isinstance(model, dict):
                    model_id = model.get('id')
                elif isinstance(model, (list, tuple)):
                    model_id = model[0]
                else:
                    model_id = getattr(model, 'id', None)
                
                if not isinstance(model_id, str):
                    continue
                    
                # Check if model matches known patterns
                if any(model_id.startswith(prefix) for prefix in known_prefixes):
                    # Prioritize latest versions and validate model tiers
                    model_lower = model_id.lower()
                    
                    # Check if it's a latest version or specific version model
                    is_latest = "latest" in model_lower
                    has_valid_tier = any(tier in model_lower for tier in model_tiers)
                    
                    if is_latest or has_valid_tier:
                        filtered_models.append(model_id)
                        
                        # If we found a specific version, also add its latest counterpart
                        if not is_latest and has_valid_tier:
                            # Extract the model tier
                            tier = next(tier for tier in model_tiers if tier in model_lower)
                            latest_version = f"mistral-{tier}-latest"
                            if latest_version not in filtered_models:
                                filtered_models.append(latest_version)
            except Exception as e:
                logging.warning(f"Error processing model: {str(e)}")
                continue
        
        return sorted(filtered_models) if filtered_models else self._available_models

    def set_api_key(self, api_key: str) -> List[str]:
        """Set the Mistral API key, initialize client and fetch available models."""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key provided")
            
        # Check if we can use cached models
        if (self._api_key == api_key and self._is_cache_valid() and 
            self._cached_models is not None):
            return self._cached_models
        
        try:
            # Initialize client
            self._api_key = api_key
            self._client = Mistral(api_key=api_key)
            
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Test API key validity with a simple models list request
                    response = self._client.models.list()
                    
                    # Handle different response formats
                    if isinstance(response, (list, tuple)):
                        models_data = response
                    else:
                        models_data = getattr(response, 'data', [])

                    if not models_data:
                        print("Warning: Empty response from Mistral API, using default models")
                        self._cached_models = self._available_models
                        self._cache_timestamp = datetime.now()
                        return self._available_models
                        
                    # Filter and process models
                    filtered_models = self._filter_models(models_data)
                    
                    # Update cache
                    self._cached_models = filtered_models
                    self._cache_timestamp = datetime.now()
                    
                    print(f"Available Mistral models: {filtered_models}")
                    return filtered_models
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    break
                    
            # If we get here, all retries failed
            error_msg = str(last_error).lower()
            if "timeout" in error_msg:
                raise ValueError("Mistral API timeout. Please try again later.")
            elif "401" in error_msg or "unauthorized" in error_msg:
                raise ValueError("Invalid API key")
            elif "404" in error_msg:
                raise ValueError("API endpoint not found. Please check your Mistral API configuration.")
            elif "rate" in error_msg:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif "connection" in error_msg:
                raise ValueError("Connection error. Please check your internet connection.")
            else:
                raise ValueError(f"Error connecting to Mistral: {str(last_error)}")
                    
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                raise ValueError("Mistral API timeout. Please try again later.")
            elif "401" in error_msg or "unauthorized" in error_msg:
                raise ValueError("Invalid API key")
            elif "404" in error_msg:
                raise ValueError("API endpoint not found. Please check your Mistral API configuration.")
            elif "rate" in error_msg:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif "connection" in error_msg:
                raise ValueError("Connection error. Please check your internet connection.")
            else:
                raise ValueError(f"Error connecting to Mistral: {str(e)}")

    def compare_texts(self, student_text: str, solution_text: str, 
                     model_name: str = "mistral-medium-latest", 
                     marking_criteria: Optional[str] = None,
                     temperature: float = 0.1,
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using Mistral."""
        if not self._client:
            return {
                "detailed_feedback": "Error: Mistral API key not set. Please provide your API key in the settings."
            }

        # Validate model name and find best match if needed
        if not isinstance(model_name, str):
            model_name = self._available_models[0]
            print(f"Invalid model name type, falling back to {model_name}")
        elif model_name not in self.available_models:
            # Try to find closest match based on prefix and tier
            model_parts = model_name.split('-')
            if len(model_parts) > 1:
                matching_models = [
                    m for m in self.available_models 
                    if model_parts[0] in m and model_parts[1] in m
                ]
                if matching_models:
                    model_name = matching_models[0]
                    print(f"Model '{model_name}' not found, using similar model: {model_name}")
                else:
                    model_name = self._available_models[0]
                    print(f"No similar model found, falling back to {model_name}")
            else:
                model_name = self._available_models[0]
                print(f"Invalid model format, falling back to {model_name}")

        try:
            criteria_section = ""
            if marking_criteria and marking_criteria.strip():
                criteria_section = f"Use these marking criteria: {marking_criteria}"

            prompt = f'''Grade the student submission by comparing it with the solution. Provide clear feedback with scores and comments for each question.
For each question, include the score in this format: [Score: X/Y]
Include a total score at the end.

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
                    # Get response from Mistral with natural language format
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a precise grading assistant. Provide clear, structured feedback with specific scores for each question and an overall score."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    
                    chat_response = self._client.chat.complete(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Extract response text directly
                    if chat_response and hasattr(chat_response, 'choices') and chat_response.choices:
                        response_text = chat_response.choices[0].message.content
                        return {'detailed_feedback': response_text}
                    else:
                        raise ValueError("Empty or invalid response from Mistral")
                        
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    break
            
            # If we get here, all retries failed
            error_msg = str(last_error).lower()
            if "timeout" in error_msg:
                return {
                    "detailed_feedback": "Error: Request timed out. Please try again."
                }
            elif "404" in error_msg:
                return {
                    "detailed_feedback": f"Error: Invalid model '{model_name}'. Please select a different model."
                }
            elif "rate" in error_msg:
                return {
                    "detailed_feedback": "Error: Rate limit exceeded. Please try again later."
                }
            return {
                "detailed_feedback": f"Error processing submission: {error_msg}"
            }
                
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                return {
                    "detailed_feedback": "Error: Request timed out. Please try again."
                }
            elif "404" in error_msg:
                return {
                    "detailed_feedback": f"Error: Invalid model '{model_name}'. Please select a different model."
                }
            elif "rate" in error_msg:
                return {
                    "detailed_feedback": "Error: Rate limit exceeded. Please try again later."
                }
            return {
                "detailed_feedback": f"Error processing submission: {error_msg}"
            }
