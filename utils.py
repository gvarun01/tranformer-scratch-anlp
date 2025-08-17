# utils.py - Helper functions for Transformer from scratch
# This file will contain utility functions without using torch.nn.Transformer modules

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import re
import unicodedata
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

class CustomLayerNorm(nn.Module):
    """Custom LayerNorm implementation (no torch.nn.LayerNorm)"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small value to prevent division by zero
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters: gamma (scale) and beta (shift)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean and variance along the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters: gamma * normalized + beta
        output = self.gamma * normalized + self.beta
        
        return output

class RoPE(nn.Module):
    """Rotary Positional Embeddings (RoPE) for positional encoding"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            d_model: Model dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency calculation
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create position embeddings
        self._create_position_embeddings()
    
    def _create_position_embeddings(self):
        """Create position embeddings for RoPE"""
        # Create position indices
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        
        # Create frequency bands: d_model/2 frequency bands
        freqs = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model))
        
        # Create position-frequency matrix
        # Shape: (max_seq_len, d_model//2)
        pos_freqs = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Create sin and cos embeddings
        # Shape: (max_seq_len, d_model)
        sin_embeddings = torch.sin(pos_freqs)
        cos_embeddings = torch.cos(pos_freqs)
        
        # Interleave sin and cos embeddings
        # [sin_0, cos_0, sin_1, cos_1, ...]
        self.register_buffer('sin_embeddings', sin_embeddings)
        self.register_buffer('cos_embeddings', cos_embeddings)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions for RoPE"""
        # Split tensor into two halves along last dimension
        x1, x2 = x.chunk(2, dim=-1)
        
        # Rotate: [x1, x2] -> [-x2, x1]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            seq_len: Optional sequence length (if different from x.size(1))
        
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Ensure sequence length doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # Get sin and cos embeddings for current sequence length
        sin_emb = self.sin_embeddings[:seq_len]  # (seq_len, d_model//2)
        cos_emb = self.cos_embeddings[:seq_len]  # (seq_len, d_model//2)
        
        # Expand to match input dimensions
        # (seq_len, d_model//2) -> (1, seq_len, d_model//2)
        sin_emb = sin_emb.unsqueeze(0)
        cos_emb = cos_emb.unsqueeze(0)
        
        # Interleave sin and cos to match d_model dimension
        # [sin_0, cos_0, sin_1, cos_1, ...]
        sin_interleaved = sin_emb.repeat_interleave(2, dim=-1)
        cos_interleaved = cos_emb.repeat_interleave(2, dim=-1)
        
        # Apply RoPE: x * cos + rotate_half(x) * sin
        rotated = x * cos_interleaved + self._rotate_half(x) * sin_interleaved
        
        return rotated
    
    def apply_rope_to_qk(self, Q: torch.Tensor, K: torch.Tensor, 
                         seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Query and Key tensors
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            seq_len: Optional sequence length
        
        Returns:
            Tuple of (Q_with_rope, K_with_rope)
        """
        batch_size, num_heads, seq_len_q, d_k = Q.shape
        _, _, seq_len_k, _ = K.shape
        
        # Reshape to apply RoPE: (batch_size * num_heads, seq_len, d_k)
        Q_reshaped = Q.view(-1, seq_len_q, d_k)
        K_reshaped = K.view(-1, seq_len_k, d_k)
        
        # Apply RoPE
        Q_rope = self.forward(Q_reshaped, seq_len_q)
        K_rope = self.forward(K_reshaped, seq_len_k)
        
        # Reshape back to original shape
        Q_rope = Q_rope.view(batch_size, num_heads, seq_len_q, d_k)
        K_rope = K_rope.view(batch_size, num_heads, seq_len_k, d_k)
        
        return Q_rope, K_rope

class RelativePositionBias(nn.Module):
    """Relative Position Bias for positional encoding"""
    
    def __init__(self, max_seq_len: int = 2048, num_heads: int = 8, 
                 max_relative_distance: int = 128):
        """
        Args:
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            max_relative_distance: Maximum relative distance to consider
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.max_relative_distance = max_relative_distance
        
        # Create relative position bias table
        # Shape: (2 * max_relative_distance + 1, num_heads)
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_distance + 1, num_heads
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize relative position bias weights"""
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative position indices for a given sequence length"""
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.long)
        
        # Create relative position matrix
        # Shape: (seq_len, seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip relative positions to max_relative_distance
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_distance, 
            self.max_relative_distance
        )
        
        # Shift to non-negative indices for embedding lookup
        relative_positions += self.max_relative_distance
        
        return relative_positions
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position bias for attention scores
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Relative position bias tensor of shape (1, num_heads, seq_len, seq_len)
        """
        # Get relative position indices
        relative_positions = self._get_relative_positions(seq_len)
        
        # Lookup relative position bias
        # Shape: (seq_len, seq_len, num_heads)
        bias = self.relative_attention_bias(relative_positions)
        
        # Transpose to match attention score shape
        # Shape: (1, num_heads, seq_len, seq_len)
        bias = bias.transpose(0, 2).unsqueeze(0)
        
        return bias
    
    def add_to_attention_scores(self, attention_scores: torch.Tensor, 
                               seq_len: int) -> torch.Tensor:
        """
        Add relative position bias to attention scores
        
        Args:
            attention_scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
            seq_len: Sequence length
        
        Returns:
            Attention scores with relative position bias added
        """
        # Get relative position bias
        relative_bias = self.forward(seq_len)
        
        # Add bias to attention scores
        # relative_bias: (1, num_heads, seq_len, seq_len)
        # attention_scores: (batch_size, num_heads, seq_len, seq_len)
        scores_with_bias = attention_scores + relative_bias
        
        return scores_with_bias

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention mechanism
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        V: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
        mask: Optional mask tensor to mask out certain positions
    
    Returns:
        Tuple of (output, attention_weights)
        - output: shape (batch_size, num_heads, seq_len, d_k)
        - attention_weights: shape (batch_size, num_heads, seq_len, seq_len)
    """
    # Get dimensions
    batch_size, num_heads, seq_len, d_k = Q.shape
    
    # Compute attention scores: Q * K^T
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided (True values are masked out)
    if mask is not None:
        # Expand mask to match scores shape if needed
        if mask.dim() == 2:  # (seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        elif mask.dim() == 3:  # (batch_size, 1, seq_len)
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        
        # Apply mask: set masked positions to large negative value
        scores = scores.masked_fill(mask, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values: attention_weights * V
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

class TextPreprocessor:
    """Text preprocessing pipeline for Finnish-English translation"""
    
    def __init__(self):
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and normalizing unicode"""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing"""
        # Normalize text
        text = self.normalize_text(text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[''′]', "'", text)
        text = re.sub(r'[""″]', '"', text)
        
        # Normalize dashes
        text = re.sub(r'[–—−]', '-', text)
        
        return text
    
    def add_special_tokens(self, text: str, add_bos: bool = True, add_eos: bool = True) -> str:
        """Add special tokens to text"""
        tokens = []
        
        if add_bos:
            tokens.append(BOS_TOKEN)
        
        tokens.append(text)
        
        if add_eos:
            tokens.append(EOS_TOKEN)
        
        return " ".join(tokens)
    
    def preprocess_text(self, text: str, add_special_tokens: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            cleaned_text = self.add_special_tokens(cleaned_text)
        
        return cleaned_text

class Vocabulary:
    """Vocabulary management for tokenization"""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2id = {}
        self.id2token = {}
        self.token_freq = {}
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        
        # Initialize special tokens
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token
    
    def add_token(self, token: str, freq: int = 1):
        """Add token to vocabulary with frequency"""
        if token in self.token_freq:
            self.token_freq[token] += freq
        else:
            self.token_freq[token] = freq
    
    def build_vocab(self, tokenized_texts: List[List[str]]):
        """Build vocabulary from tokenized texts"""
        # Count token frequencies
        for tokens in tokenized_texts:
            for token in tokens:
                self.add_token(token)
        
        # Filter by minimum frequency and add to vocabulary
        vocab_tokens = [token for token, freq in self.token_freq.items() 
                       if freq >= self.min_freq or token in self.special_tokens]
        
        # Sort by frequency (descending) then alphabetically
        vocab_tokens.sort(key=lambda x: (-self.token_freq.get(x, 0), x))
        
        # Build mappings
        for i, token in enumerate(vocab_tokens):
            if token not in self.token2id:  # Skip special tokens already added
                self.token2id[token] = len(self.token2id)
                self.id2token[len(self.id2token)] = token
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.token2id.get(token, self.token2id[UNK_TOKEN]) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id2token.get(id, UNK_TOKEN) for id in ids]
    
    def __len__(self) -> int:
        return len(self.token2id)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        import json
        data = {
            'token2id': self.token2id,
            'id2token': {int(k): v for k, v in self.id2token.items()},
            'token_freq': self.token_freq,
            'min_freq': self.min_freq
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from file"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(data['min_freq'])
        vocab.token2id = data['token2id']
        vocab.id2token = {int(k): v for k, v in data['id2token'].items()}
        vocab.token_freq = data['token_freq']
        
        return vocab

class TranslationDataset(Dataset):
    """Dataset for Finnish-English translation pairs"""
    
    def __init__(self, src_texts: List[str], tgt_texts: List[str], 
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, 
                 max_len: int = 512, pad_idx: int = 0):
        """
        Args:
            src_texts: List of source (Finnish) texts
            tgt_texts: List of target (English) texts
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            max_len: Maximum sequence length
            pad_idx: Padding token index
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pad_idx = pad_idx
        
        # Preprocess and tokenize all texts
        self.src_tokens = self._preprocess_texts(src_texts)
        self.tgt_tokens = self._preprocess_texts(tgt_texts)
        
        # Convert to IDs
        self.src_ids = [src_vocab.encode(tokens) for tokens in self.src_tokens]
        self.tgt_ids = [tgt_vocab.encode(tokens) for tokens in self.tgt_tokens]
    
    def _preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """Preprocess and tokenize texts"""
        processed_texts = []
        for text in texts:
            # Clean and normalize text
            cleaned = text_preprocessor.preprocess_text(text, add_special_tokens=False)
            # Simple space-based tokenization (will be replaced with SentencePiece later)
            tokens = cleaned.split()
            processed_texts.append(tokens)
        return processed_texts
    
    def _pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        """Pad sequence to max_len with pad_idx"""
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [self.pad_idx] * (max_len - len(seq))
    
    def __len__(self) -> int:
        return len(self.src_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get source and target tensors for a given index"""
        src_seq = self.src_ids[idx]
        tgt_seq = self.tgt_ids[idx]
        
        # Pad sequences
        src_padded = self._pad_sequence(src_seq, self.max_len)
        tgt_padded = self._pad_sequence(tgt_seq, self.max_len)
        
        # Convert to tensors
        src_tensor = torch.tensor(src_padded, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)
        
        return src_tensor, tgt_tensor

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for dynamic padding to batch max length
    
    Args:
        batch: List of (src_tensor, tgt_tensor) pairs
        pad_idx: Padding token index
    
    Returns:
        Batched source and target tensors with dynamic padding
    """
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    # Separate source and target sequences
    src_sequences = [item[0] for item in batch]
    tgt_sequences = [item[1] for item in batch]
    
    # Find max lengths in this batch
    src_max_len = max(seq.size(0) for seq in src_sequences)
    tgt_max_len = max(seq.size(0) for seq in tgt_sequences)
    
    # Pad sequences to batch max length
    src_padded = []
    tgt_padded = []
    
    for src_seq, tgt_seq in zip(src_sequences, tgt_sequences):
        # Pad source sequence
        if src_seq.size(0) < src_max_len:
            src_pad = torch.full((src_max_len - src_seq.size(0),), pad_idx, dtype=src_seq.dtype)
            src_padded.append(torch.cat([src_seq, src_pad]))
        else:
            src_padded.append(src_seq)
        
        # Pad target sequence
        if tgt_seq.size(0) < tgt_max_len:
            tgt_pad = torch.full((tgt_max_len - tgt_seq.size(0),), pad_idx, dtype=tgt_seq.dtype)
            tgt_padded.append(torch.cat([tgt_seq, tgt_pad]))
        else:
            tgt_padded.append(tgt_seq)
    
    # Stack into batch tensors
    src_batch = torch.stack(src_padded)
    tgt_batch = torch.stack(tgt_padded)
    
    return src_batch, tgt_batch

def create_dataloader(dataset: TranslationDataset, batch_size: int, shuffle: bool = True, 
                     num_workers: int = 0, pad_idx: int = 0) -> DataLoader:
    """Create DataLoader with custom collate function for dynamic padding
    
    Args:
        dataset: TranslationDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pad_idx: Padding token index
    
    Returns:
        DataLoader with dynamic padding
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        drop_last=False
    )

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Create padding mask to mask out pad tokens
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_idx: Index of padding token
    
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len) where True means mask out
    """
    # Create mask: True where seq == pad_idx (should be masked)
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask

def create_look_ahead_mask(size: int) -> torch.Tensor:
    """Create look-ahead mask to prevent decoder from attending to future tokens
    
    Args:
        size: Size of the sequence
    
    Returns:
        Upper triangular mask tensor of shape (size, size) where True means mask out
    """
    # Create upper triangular matrix (including diagonal)
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    
    # Convert to boolean: True means mask out (should not attend)
    mask = mask.bool()
    
    return mask

def create_combined_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Create combined mask for decoder (padding + look-ahead)
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_idx: Index of padding token
    
    Returns:
        Combined mask tensor of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_len = seq.shape
    
    # Create padding mask
    padding_mask = create_padding_mask(seq, pad_idx)  # (batch_size, 1, 1, seq_len)
    
    # Create look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)  # (seq_len, seq_len)
    
    # Expand look-ahead mask to batch dimension
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    # Combine masks: True means mask out (should not attend)
    combined_mask = padding_mask | look_ahead_mask
    
    return combined_mask

# Initialize global preprocessor
text_preprocessor = TextPreprocessor()
