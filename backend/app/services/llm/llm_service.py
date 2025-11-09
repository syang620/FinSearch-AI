from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, Any
from app.core.config import settings
import torch
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM models (Flan-T5 or Mistral-7B)"""

    def __init__(self):
        self.model_name = settings.LLM_MODEL_NAME
        self.device = settings.LLM_DEVICE
        self.max_length = settings.LLM_MAX_LENGTH
        self.temperature = settings.LLM_TEMPERATURE
        self.model = None
        self.tokenizer = None
        self.is_seq2seq = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLM model and tokenizer"""
        try:
            logger.info(f"Loading LLM model: {self.model_name}")

            # Determine model type
            self.is_seq2seq = "t5" in self.model_name.lower() or "flan" in self.model_name.lower()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model based on type
            if self.is_seq2seq:
                # Seq2Seq model (Flan-T5)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device if self.device == "cuda" else None
                )
            else:
                # Causal LM model (Mistral)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device if self.device == "cuda" else None
                )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info(f"LLM model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            max_new_tokens = max_new_tokens or self.max_length
            temperature = temperature or self.temperature

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                if self.is_seq2seq:
                    # Seq2Seq generation (Flan-T5)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        **kwargs
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Causal LM generation (Mistral)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
                    # Only return the newly generated text (remove prompt)
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def chat(
        self,
        query: str,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Chat interface for the LLM

        Args:
            query: User query
            context: Optional context (from RAG)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response
        """
        try:
            # Build prompt
            if context:
                prompt = f"""Context: {context}

Question: {query}

Answer:"""
            else:
                prompt = f"""Question: {query}

Answer:"""

            # Generate response
            response = self.generate(prompt, max_new_tokens=max_new_tokens)

            return response

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_type": "seq2seq" if self.is_seq2seq else "causal_lm",
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
        }


# Singleton instance
llm_service = LLMService()
