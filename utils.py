# utils.py - Helper functions for Transformer from scratch
# This file will contain utility functions without using torch.nn.Transformer
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
        
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        seq_len = x.size(2)
        
        # Ensure sequence length doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # Get sin and cos embeddings for current sequence length
        sin_emb = self.sin_embeddings[:seq_len, :]  # (seq_len, d_model//2)
        cos_emb = self.cos_embeddings[:seq_len, :]  # (seq_len, d_model//2)
        
        # Expand to match input dimensions
        # (seq_len, d_model//2) -> (1, 1, seq_len, d_model//2)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(1)
        
        # Interleave sin and cos to match d_k dimension
        # [sin_0, cos_0, sin_1, cos_1, ...]
        sin_interleaved = sin_emb.repeat_interleave(2, dim=-1)
        cos_interleaved = cos_emb.repeat_interleave(2, dim=-1)
        
        # Apply RoPE: x * cos + rotate_half(x) * sin
        rotated = x * cos_interleaved + self._rotate_half(x) * sin_interleaved
        
        return rotated
    
    def get_embeddings(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for a given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            Tuple of (cos, sin) embeddings of shape (1, 1, seq_len, d_model)
        """
        # Ensure sequence length doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # Get sin and cos embeddings for current sequence length
        sin_emb = self.sin_embeddings[:seq_len, :]  # (seq_len, d_model//2)
        cos_emb = self.cos_embeddings[:seq_len, :]  # (seq_len, d_model//2)
        
        # Expand to match input dimensions
        # (seq_len, d_model//2) -> (1, 1, seq_len, d_model//2)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(1)
        
        # Interleave sin and cos to match d_model dimension
        # [sin_0, cos_0, sin_1, cos_1, ...]
        sin_interleaved = sin_emb.repeat_interleave(2, dim=-1)
        cos_interleaved = cos_emb.repeat_interleave(2, dim=-1)
        
        return cos_interleaved.to(device), sin_interleaved.to(device)
    
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
    """Relative Position Bias for Transformer attention"""
    
    def __init__(self, max_seq_len: int, num_heads: int):
        """
        Args:
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        
        # Create learnable embeddings for relative positions
        # The range of relative positions is [-max_seq_len+1, max_seq_len-1]
        num_relative_positions = 2 * max_seq_len - 1
        self.relative_attention_bias = nn.Embedding(num_relative_positions, num_heads)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.xavier_uniform_(self.relative_attention_bias.weight)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Create the bias tensor for a given sequence length.
        
        Args:
            seq_len: The sequence length of the input.
            
        Returns:
            A bias tensor of shape (1, num_heads, seq_len, seq_len)
        """
        # Create a matrix of relative positions
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Shift relative positions to be non-negative (0 to 2*seq_len-2)
        relative_positions = relative_positions + seq_len - 1
        
        # Get the bias from the embedding table
        # Shape: (seq_len, seq_len, num_heads)
        bias = self.relative_attention_bias(relative_positions)
        
        # Reshape for broadcasting with attention scores
        # (seq_len, seq_len, num_heads) -> (num_heads, seq_len, seq_len) -> (1, num_heads, seq_len, seq_len)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
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
                                 mask: Optional[torch.Tensor] = None,
                                 relative_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention mechanism
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        V: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
        mask: Optional mask tensor to mask out certain positions
        relative_bias: Optional relative position bias tensor
    
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

    # Add relative position bias if provided
    if relative_bias is not None:
        scores += relative_bias
    
    # Apply mask if provided (True values are masked out)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
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
                 src_vocab, tgt_vocab, 
                 max_len: int = 512, pad_idx: int = 0):
        """
        Args:
            src_texts: List of source (Finnish) texts
            tgt_texts: List of target (English) texts
            src_vocab: Source vocabulary (can be Vocabulary or SPMWrapper)
            tgt_vocab: Target vocabulary (can be Vocabulary or SPMWrapper)
            max_len: Maximum sequence length
            pad_idx: Padding token index
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pad_idx = pad_idx
        
        # Check if using SentencePiece (SPMWrapper) or traditional vocabulary
        self.use_spm = hasattr(src_vocab, 'sp')
        
        if self.use_spm:
            # For SentencePiece, encode texts directly
            self.src_ids = [src_vocab.encode(text.strip()) for text in src_texts]
            self.tgt_ids = [tgt_vocab.encode(text.strip()) for text in tgt_texts]
        else:
            # For traditional vocabulary, preprocess and tokenize first
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
        Mask tensor of shape (batch_size, 1, 1, seq_len) where False means mask out
    """
    # Create mask: False where seq == pad_idx (should be masked), True where valid
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask

def create_look_ahead_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create a look-ahead mask for decoding.
    
    Args:
        size: The sequence length.
        device: The device to create the tensor on.
        
    Returns:
        A look-ahead mask tensor of shape (1, 1, size, size) where False means mask out.
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask == 0  # True for valid positions, False for future positions

def create_combined_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a combined mask for the decoder, including padding and look-ahead.
    
    Args:
        seq: The target sequence tensor of shape (batch_size, seq_len).
        pad_idx: The padding token index.
        
    Returns:
        A combined mask tensor where False means mask out.
    """
    seq_len = seq.shape[1]
    padding_mask = create_padding_mask(seq, pad_idx)  # (batch_size, 1, 1, seq_len)
    look_ahead_mask = create_look_ahead_mask(seq_len, seq.device)  # (seq_len, seq_len)
    
    # Combine masks for decoder self-attention
    # A position can be attended to if it is not a padding token AND not a future token.
    # The look_ahead_mask is broadcast across the batch dimension.
    combined_mask = padding_mask & look_ahead_mask.unsqueeze(0)
    return combined_mask

# Initialize global preprocessor
text_preprocessor = TextPreprocessor()

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function to rotate half of the dimensions"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to a tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, d_k).
        cos (torch.Tensor): Cosine part of the embedding of shape (1, 1, max_seq_len, d_k).
        sin (torch.Tensor): Sine part of the embedding of shape (1, 1, max_seq_len, d_k).
        
    Returns:
        torch.Tensor: Tensor with rotary positional embedding applied.
    """
    # Get seq_len from input tensor
    seq_len = x.shape[2]
    
    # Adjust cos and sin tensors to match the sequence length of the input
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]

    # Apply rotary embeddings
    x_rotated = _rotate_half(x)
    return x * cos + x_rotated * sin

def greedy_decode(model, src_tensor, src_mask, max_len, start_symbol, end_symbol, device):
    """
    Greedy decoding for translation.
    """
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)
        # Start with the start-of-sequence token
        tgt_tokens = torch.full((src_tensor.size(0), 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = create_combined_mask(tgt_tokens, model.pad_idx).to(device)
            
            # Create cross-attention mask by expanding src_mask to target length
            # src_mask: (batch_size, 1, 1, src_len)
            # We need: (batch_size, 1, tgt_len, src_len)
            tgt_len = tgt_tokens.size(1)
            cross_attn_mask = src_mask.expand(-1, -1, tgt_len, -1)
            
            output = model.decoder(tgt_tokens, encoder_output, tgt_mask, cross_attn_mask)
            
            # Get the token with the highest probability
            pred_token = output.argmax(2)[:, -1]
            
            # Append the predicted token
            tgt_tokens = torch.cat((tgt_tokens, pred_token.unsqueeze(1)), dim=1)

            # Stop if all sequences have generated the end token
            if (pred_token == end_symbol).all():
                break
                
    return tgt_tokens

def beam_search_decode(model, src_tensor, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """
    Beam search decoding for translation.
    """
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)
        
        # Initialize beams
        # Each beam is a tuple of (sequence, score)
        beams = [(torch.full((1, 1), start_symbol, dtype=torch.long, device=device), 0.0)]
        
        for _ in range(max_len - 1):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1] == end_symbol:
                    new_beams.append((seq, score))
                    continue

                tgt_mask = model.create_tgt_mask(seq).to(device)
                output = model.decoder(seq, encoder_output, tgt_mask, src_mask)
                
                # Get the log probabilities of the next token
                log_probs = F.log_softmax(output[:, -1], dim=-1)
                top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width, dim=-1)

                for i in range(beam_width):
                    new_seq = torch.cat([seq, top_k_tokens[:, i].unsqueeze(1)], dim=1)
                    new_score = score + top_k_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))
            
            # Sort all new beams by score and keep the top `beam_width`
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if the top beam has ended
            if beams[0][0][0, -1] == end_symbol:
                break
                
    return beams[0][0]

def top_k_sampling_decode(model, src_tensor, src_mask, max_len, start_symbol, end_symbol, device, k=5):
    """
    Top-k sampling for translation.
    """
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)
        tgt_tokens = torch.full((src_tensor.size(0), 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = create_combined_mask(tgt_tokens, model.pad_idx).to(device)
            output = model.decoder(tgt_tokens, encoder_output, tgt_mask, src_mask)
            
            # Get the logits for the last token
            last_logits = output[:, -1, :]
            
            # Get top-k logits and their indices
            top_k_logits, top_k_indices = torch.topk(last_logits, k, dim=-1)
            
            # Apply softmax to the top-k logits to get probabilities
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the distribution
            sampled_index = torch.multinomial(top_k_probs, num_samples=1)
            
            # Get the actual token index
            pred_token = torch.gather(top_k_indices, -1, sampled_index)
            
            # Append the predicted token
            tgt_tokens = torch.cat((tgt_tokens, pred_token), dim=1)

            # Stop if the end token is generated
            if pred_token.item() == end_symbol:
                break
                
    return tgt_tokens
