"""
Embedding module for financial documents.
Supports multiple embedding providers: Ollama, Abaci FinE5, and SentenceTransformers.
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ========= Base Classes =========

@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict


class EmbedderInterface(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get embedder name"""
        pass


# ========= Ollama Embedder =========

class OllamaEmbedder(EmbedderInterface):
    """Embedder using Ollama's local models with parallel processing"""

    def __init__(self, model_name: str = "nomic-embed-text", max_workers: int = 8):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/embeddings"
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._dimension = None

        # Test connection
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": "test"},
                timeout=10
            )
            if response.status_code == 200:
                test_embedding = response.json()["embedding"]
                self._dimension = len(test_embedding)
                print(f"✓ Connected to Ollama model: {model_name}")
                print(f"  Embedding dimension: {self._dimension}")
            else:
                raise Exception(f"Failed to connect to Ollama: {response.text}")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama. Make sure ollama is running: {e}")

    def _embed_single(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                return None
        except Exception:
            return None

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using parallel processing"""
        embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._embed_single, text): i
                for i, text in enumerate(texts)
            }

            with tqdm(total=len(texts), desc=f"Generating {self.model_name} embeddings") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        embedding = future.result()
                        if embedding:
                            embeddings[index] = embedding
                        else:
                            # Retry once if failed
                            embedding = self._embed_single(texts[index])
                            embeddings[index] = embedding
                    except Exception:
                        embeddings[index] = None
                    pbar.update(1)

        # Filter out any None values
        valid_embeddings = [e for e in embeddings if e is not None]
        if len(valid_embeddings) < len(texts):
            print(f"Warning: Only {len(valid_embeddings)}/{len(texts)} embeddings succeeded")

        return embeddings

    def get_dimension(self) -> int:
        return self._dimension

    def get_name(self) -> str:
        return f"ollama/{self.model_name}"


# ========= Abaci FinE5 Embedder =========

class FinE5Embedder(EmbedderInterface):
    """Embedder using Abaci FinE5 API for financial documents"""

    ABACI_API_URL = "https://abacinlp.com/v1/embeddings"
    MODEL_NAME = "abacinlp-text-v1"
    DIMENSION = 4096  # FinE5 embedding dimension

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0,
                 use_instruction: bool = True, task_description: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ABACI_API_KEY")
        if not self.api_key:
            raise ValueError("ABACI_API_KEY not provided. Set via parameter or environment variable.")

        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.use_instruction = use_instruction
        self.task_description = task_description or "Given a financial document, retrieve relevant information that answers financial questions."

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        print(f"✓ Initialized FinE5 embedder")
        print(f"  Model: {self.MODEL_NAME}")
        print(f"  Dimension: {self.DIMENSION}")
        print(f"  Instruction prompting: {self.use_instruction}")

    def _format_text_with_instruction(self, text: str) -> str:
        """Format text with instruction prompting for better retrieval"""
        if not self.use_instruction:
            return text
        return f'Instruct: {self.task_description}\nQuery: {text}'

    def embed_batch(self, texts: List[str], retry_count: int = 3) -> List[List[float]]:
        """Generate embeddings via FinE5 API with retry logic"""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        # Apply instruction prompting if enabled
        formatted_texts = [self._format_text_with_instruction(t) for t in texts] if self.use_instruction else texts

        payload = {
            "model": self.MODEL_NAME,
            "input": formatted_texts,
            "encoding_format": "float"
        }

        for attempt in range(retry_count):
            try:
                response = self.session.post(
                    self.ABACI_API_URL,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                self.last_request_time = time.time()

                data = response.json()

                # Extract embeddings from response
                embeddings = []
                for item in sorted(data["data"], key=lambda x: x["index"]):
                    embeddings.append(item["embedding"])

                if len(embeddings) != len(texts):
                    raise ValueError(f"Mismatch: requested {len(texts)}, got {len(embeddings)}")

                return embeddings

            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise e

        return []

    def get_dimension(self) -> int:
        return self.DIMENSION

    def get_name(self) -> str:
        return "abaci/FinE5"


# ========= SentenceTransformer Embedder =========

class SentenceTransformerEmbedder(EmbedderInterface):
    """Embedder using SentenceTransformers models (GPU-optimized)"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        # Use GPU if available (CUDA or MPS for Apple Silicon)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)

        # Optimize for GPU (fp16 for CUDA only, MPS doesn't support half precision well)
        if device == "cuda":
            self.model.half()

        self._dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = 32 if device in ["cuda", "mps"] else 8

        print(f"✓ Loaded {model_name} on {device}")
        print(f"  Dimension: {self._dimension}")
        print(f"  Batch size: {self.batch_size}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with GPU optimization"""
        import torch

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )

        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self._dimension

    def get_name(self) -> str:
        return f"sentence-transformers/{self.model_name}"


# ========= NV-Embed-v2 Embedder =========

class NVEmbedV2Embedder(EmbedderInterface):
    """
    Embedder using NVIDIA's NV-Embed-v2 model.
    State-of-the-art embedding model optimized for retrieval tasks.
    Works with MPS (Apple Silicon), CUDA, or CPU.
    """

    MODEL_NAME = "nvidia/NV-Embed-v2"
    DIMENSION = 4096  # NV-Embed-v2 produces 4096-dim embeddings

    def __init__(self, device: Optional[str] = None, trust_remote_code: bool = True):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # Determine best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading NV-Embed-v2...")
        print(f"  Device: {device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=trust_remote_code
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=trust_remote_code
        ).to(device)

        self.model.eval()  # Set to evaluation mode

        # Set batch size based on device
        if device == "mps":
            self.batch_size = 8  # MPS can handle moderate batches
        elif device == "cuda":
            self.batch_size = 16  # CUDA can handle larger batches
        else:
            self.batch_size = 4  # CPU is slower

        print(f"✓ Loaded NV-Embed-v2 on {device}")
        print(f"  Dimension: {self.DIMENSION}")
        print(f"  Batch size: {self.batch_size}")
        if device == "mps":
            print(f"  Using Apple Silicon GPU acceleration!")

    def _add_eos_token(self, texts: List[str]) -> List[str]:
        """Add EOS token as required by NV-Embed-v2"""
        eos = self.tokenizer.eos_token
        return [text + eos for text in texts]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with MPS/CUDA/CPU support"""
        import torch

        # Add EOS token to all texts
        texts_with_eos = self._add_eos_token(texts)

        all_embeddings = []

        # Process in sub-batches
        with torch.no_grad():
            for i in range(0, len(texts_with_eos), self.batch_size):
                batch_texts = texts_with_eos[i:i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings
                outputs = self.model(**inputs)

                # Use mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state

                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                # Normalize
                mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

                all_embeddings.extend(mean_embeddings.cpu().numpy().tolist())

        return all_embeddings

    def get_dimension(self) -> int:
        return self.DIMENSION

    def get_name(self) -> str:
        return "nvidia/NV-Embed-v2"


# ========= Factory Function =========

def create_embedder(provider: str = "ollama", **kwargs) -> EmbedderInterface:
    """
    Factory function to create embedder instances.

    Args:
        provider: One of "ollama", "fine5", "sentence-transformers", or "nv-embed-v2"
        **kwargs: Provider-specific arguments
            - For "ollama": model_name, max_workers
            - For "fine5": api_key, rate_limit_delay, use_instruction, task_description
            - For "sentence-transformers": model_name, device
            - For "nv-embed-v2": device, trust_remote_code

    Returns:
        EmbedderInterface instance

    Examples:
        embedder = create_embedder("ollama", model_name="nomic-embed-text")
        embedder = create_embedder("fine5", api_key="sk-...", use_instruction=True)
        embedder = create_embedder("fine5", api_key="sk-...",
                                   task_description="Given a financial question, retrieve relevant answers.")
        embedder = create_embedder("sentence-transformers", model_name="BAAI/bge-large-en-v1.5")
        embedder = create_embedder("nv-embed-v2")  # Auto-detects MPS/CUDA/CPU
    """
    providers = {
        "ollama": OllamaEmbedder,
        "fine5": FinE5Embedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "st": SentenceTransformerEmbedder,  # alias
        "nv-embed-v2": NVEmbedV2Embedder,
        "nvembed": NVEmbedV2Embedder,  # alias
    }

    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")

    return providers[provider.lower()](**kwargs)


# ========= Batch Processing Utilities =========

def process_chunks_with_embeddings(
    chunks: List[Dict],
    embedder: EmbedderInterface,
    batch_size: int = 16
) -> List[EmbeddingResult]:
    """
    Process chunks through embedder and return results.

    Args:
        chunks: List of dicts with 'chunk_id', 'text', 'metadata'
        embedder: EmbedderInterface instance
        batch_size: Number of chunks per batch

    Returns:
        List of EmbeddingResult objects
    """
    results = []

    print(f"\nProcessing {len(chunks)} chunks with {embedder.get_name()}")
    print(f"Batch size: {batch_size}")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        # Get embeddings
        embeddings = embedder.embed_batch(texts)

        # Create results
        for chunk, embedding in zip(batch, embeddings):
            if embedding is not None:
                result = EmbeddingResult(
                    id=chunk.get("chunk_id", chunk.get("id")),
                    text=chunk["text"],
                    embedding=embedding,
                    metadata=chunk.get("metadata", {})
                )
                results.append(result)

    print(f"✓ Processed {len(results)} chunks successfully")
    return results


def save_embeddings_to_jsonl(results: List[EmbeddingResult], output_path: str):
    """Save embedding results to JSONL file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            record = {
                "id": result.id,
                "text": result.text,
                "embedding": result.embedding,
                "metadata": result.metadata
            }
            f.write(json.dumps(record) + '\n')

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(results)} embeddings to {output_path} ({file_size:.2f} MB)")


def load_embeddings_from_jsonl(input_path: str) -> List[EmbeddingResult]:
    """Load embedding results from JSONL file"""
    results = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            result = EmbeddingResult(
                id=data["id"],
                text=data["text"],
                embedding=data["embedding"],
                metadata=data["metadata"]
            )
            results.append(result)

    print(f"✓ Loaded {len(results)} embeddings from {input_path}")
    return results