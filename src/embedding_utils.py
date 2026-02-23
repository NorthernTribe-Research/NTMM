"""
Medical Text Embedding Utilities for NTMM

Generate and manage embeddings for medical text using NTMM models.
Useful for:
- Semantic search
- Clustering medical documents
- Similarity computation
- Retrieval-augmented generation (RAG)

Copyright (c) 2026 NorthernTribe Research
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MedicalEmbeddingGenerator:
    """
    Generate embeddings from NTMM models for medical text.
    
    Features:
    - Mean pooling for sentence embeddings
    - CLS token embeddings
    - Max pooling
    - Batch processing
    - Similarity computation
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        pooling_strategy: str = "mean",
        normalize: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_path: Path to NTMM model
            pooling_strategy: "mean", "cls", or "max"
            normalize: Whether to L2-normalize embeddings
            device: Device to use (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Pooling strategy: {self.pooling_strategy}")
        print(f"Normalization: {self.normalize}")
    
    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over sequence dimension with attention mask.
        
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq]
            
        Returns:
            Pooled embeddings [batch, hidden]
        """
        # Expand attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # Sum mask
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean
        return sum_embeddings / sum_mask
    
    def _cls_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Use CLS token (first token) as embedding.
        
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq]
            
        Returns:
            CLS embeddings [batch, hidden]
        """
        return hidden_states[:, 0, :]
    
    def _max_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Max pooling over sequence dimension.
        
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq]
            
        Returns:
            Max pooled embeddings [batch, hidden]
        """
        # Set padding tokens to large negative value
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states.clone()
        hidden_states[mask_expanded == 0] = -1e9
        
        # Max pool
        return torch.max(hidden_states, dim=1)[0]
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 256,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            show_progress: Show progress bar
            
        Returns:
            Embeddings as numpy array [num_texts, hidden_dim]
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(range(num_batches), desc="Generating embeddings")
            except ImportError:
                batch_iterator = range(num_batches)
        else:
            batch_iterator = range(num_batches)
        
        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                attention_mask = encoded["attention_mask"]
                
                # Pool
                if self.pooling_strategy == "mean":
                    embeddings = self._mean_pooling(hidden_states, attention_mask)
                elif self.pooling_strategy == "cls":
                    embeddings = self._cls_pooling(hidden_states, attention_mask)
                elif self.pooling_strategy == "max":
                    embeddings = self._max_pooling(hidden_states, attention_mask)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                # Normalize
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            metric: "cosine" or "euclidean"
            
        Returns:
            Similarity matrix [len(texts1), len(texts2)]
        """
        # Generate embeddings
        emb1 = self.encode(texts1)
        emb2 = self.encode(texts2)
        
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(emb1, emb2.T)
        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            from scipy.spatial.distance import cdist
            similarity = -cdist(emb1, emb2, metric="euclidean")
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score, text) tuples
        """
        # Compute similarities
        similarities = self.compute_similarity(query, candidates)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = [
            (idx, similarities[idx], candidates[idx])
            for idx in top_indices
        ]
        
        return results
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: Union[str, Path],
        texts: Optional[List[str]] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings array
            output_path: Output file path (.npz)
            texts: Optional texts corresponding to embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if texts is not None:
            np.savez(output_path, embeddings=embeddings, texts=texts)
        else:
            np.savez(output_path, embeddings=embeddings)
        
        print(f"Saved embeddings to {output_path}")
    
    @staticmethod
    def load_embeddings(input_path: Union[str, Path]) -> tuple:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Input file path (.npz)
            
        Returns:
            (embeddings, texts) tuple (texts may be None)
        """
        data = np.load(input_path, allow_pickle=True)
        embeddings = data["embeddings"]
        texts = data.get("texts", None)
        
        return embeddings, texts


def create_embedding_index(
    texts: List[str],
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    batch_size: int = 32
):
    """
    Create an embedding index for a collection of texts.
    
    Args:
        texts: List of texts to index
        model_path: Path to NTMM model
        output_path: Output path for index
        batch_size: Batch size for processing
    """
    generator = MedicalEmbeddingGenerator(model_path)
    embeddings = generator.encode(texts, batch_size=batch_size, show_progress=True)
    generator.save_embeddings(embeddings, output_path, texts=texts)
    print(f"Created index with {len(texts)} texts")


def semantic_search(
    query: str,
    index_path: Union[str, Path],
    model_path: Union[str, Path],
    top_k: int = 5
) -> List[tuple]:
    """
    Perform semantic search on an embedding index.
    
    Args:
        query: Search query
        index_path: Path to embedding index
        model_path: Path to NTMM model
        top_k: Number of results
        
    Returns:
        List of (similarity, text) tuples
    """
    # Load index
    embeddings, texts = MedicalEmbeddingGenerator.load_embeddings(index_path)
    
    # Generate query embedding
    generator = MedicalEmbeddingGenerator(model_path)
    query_embedding = generator.encode(query)
    
    # Compute similarities
    similarities = np.dot(query_embedding, embeddings.T)[0]
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        (similarities[idx], texts[idx] if texts is not None else f"Document {idx}")
        for idx in top_indices
    ]
    
    return results
