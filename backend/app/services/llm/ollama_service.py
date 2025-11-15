"""
Ollama LLM Service for local model inference
Implements the same interface as LLMService for seamless switching
"""

import ollama
from typing import Optional, Dict, Any, List
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """Service for interacting with Ollama-hosted models"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Ollama service

        Args:
            model_name: Name of the model to use (e.g., "qwen2.5:7b", "phi3.5")
                      If not provided, uses settings.OLLAMA_MODEL
        """
        self.model_name = model_name or getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")
        self.client = ollama.Client()

        # Verify model is available
        try:
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                # Try without tag (e.g., "qwen2.5" instead of "qwen2.5:7b")
                base_name = self.model_name.split(':')[0]
                matching = [m for m in model_names if m.startswith(base_name)]
                if matching:
                    self.model_name = matching[0]
                    logger.info(f"Using model {self.model_name}")
                else:
                    raise ValueError(f"Model {self.model_name} not available. Please run: ollama pull {self.model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using Ollama model

        Args:
            prompt: The user's question or prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 = deterministic, 1.0 = creative)
            context: Optional context to include (e.g., retrieved documents)
            system_prompt: Optional system prompt to set behavior

        Returns:
            Generated text response
        """
        try:
            # Build the full prompt
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            else:
                # Default financial assistant prompt
                messages.append({
                    'role': 'system',
                    'content': (
                        "You are a knowledgeable financial analyst assistant. "
                        "Provide accurate, detailed answers about financial topics, "
                        "company analysis, and market insights. Be precise and cite "
                        "specific data when available."
                    )
                })

            # Add context if provided
            if context:
                messages.append({
                    'role': 'user',
                    'content': f"Context:\n{context}\n\nQuestion: {prompt}"
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': prompt
                })

            # Generate response
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'top_k': 40,
                }
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return f"Error generating response: {str(e)}"

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Generate text using Ollama model with streaming

        Yields:
            Chunks of generated text
        """
        try:
            # Build the full prompt
            messages = []

            # Add system prompt
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            else:
                messages.append({
                    'role': 'system',
                    'content': (
                        "You are a knowledgeable financial analyst assistant. "
                        "Provide accurate, detailed answers about financial topics."
                    )
                })

            # Add context and question
            if context:
                messages.append({
                    'role': 'user',
                    'content': f"Context:\n{context}\n\nQuestion: {prompt}"
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': prompt
                })

            # Stream response
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                }
            )

            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            yield f"Error: {str(e)}"

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text (if model supports it)
        Note: Most Ollama models don't support embeddings directly

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.warning(f"Model {self.model_name} may not support embeddings: {e}")
            # Return None or use a fallback embedding model
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama

        Returns:
            List of model information dictionaries
        """
        try:
            response = self.client.list()
            return response.get('models', [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def switch_model(self, model_name: str):
        """
        Switch to a different model

        Args:
            model_name: Name of the model to switch to
        """
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        try:
            models = self.list_models()
            for model in models:
                if model['name'] == self.model_name:
                    return {
                        'name': model['name'],
                        'size': model.get('size', 'Unknown'),
                        'modified': model.get('modified_at', 'Unknown'),
                        'family': model.get('details', {}).get('family', 'Unknown'),
                        'parameter_size': model.get('details', {}).get('parameter_size', 'Unknown'),
                        'quantization': model.get('details', {}).get('quantization_level', 'Unknown')
                    }
            return {'name': self.model_name, 'status': 'Not found'}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'name': self.model_name, 'error': str(e)}


# Singleton instance
_ollama_service = None

def get_ollama_service(model_name: Optional[str] = None) -> OllamaLLMService:
    """
    Get or create the Ollama service singleton

    Args:
        model_name: Optional model name to use

    Returns:
        OllamaLLMService instance
    """
    global _ollama_service
    if _ollama_service is None or (model_name and model_name != _ollama_service.model_name):
        _ollama_service = OllamaLLMService(model_name)
    return _ollama_service