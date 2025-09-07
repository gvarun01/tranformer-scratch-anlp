# decoder.py - Decoder classes for Transformer from scratch
# This file will contain the decoder implementation without using torch.nn.TransformerDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from utils import apply_rotary_pos_emb, scaled_dot_product_attention

class TokenEmbedding(nn.Module):
    """Token embedding layer with sqrt(d_model) scaling"""
    
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            pad_idx: Padding token index
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Set padding token embedding to zero
        if self.pad_idx is not None:
            nn.init.constant_(self.embedding.weight[self.pad_idx], 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # Get embeddings and scale by sqrt(d_model)
        embedded = self.embedding(x) * self.scale
        return embedded

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for decoder"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, pos_encoding: str = None, max_seq_len: int = 2048):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            pos_encoding: Type of positional encoding ('rope' or None)
            max_seq_len: Maximum sequence length (for RoPE)
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pos_encoding = pos_encoding
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.pos_encoding == 'rope':
            from utils import RoPE
            self.rope = RoPE(self.d_k, max_seq_len)
        elif self.pos_encoding == 'relative_bias':
            from utils import RelativePositionBias
            self.relative_bias = RelativePositionBias(max_seq_len, num_heads)
            
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layer weights"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.W_q.bias, 0.0)
        nn.init.constant_(self.W_k.bias, 0.0)
        nn.init.constant_(self.W_v.bias, 0.0)
        nn.init.constant_(self.W_o.bias, 0.0)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the heads back into d_model dimension"""
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor of shape (batch_size, seq_len, d_model)
            V: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = Q.size(0)
        
        # Linear projections and split into multiple heads
        Q = self._split_heads(self.W_q(Q))
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))
        
        if self.pos_encoding == 'rope':
            # For self-attention, Q and K have the same sequence length
            # For cross-attention, Q and K may have different sequence lengths
            q_seq_len = Q.size(2)
            k_seq_len = K.size(2)
            
            # Apply RoPE to Q with its sequence length
            q_cos, q_sin = self.rope.get_embeddings(q_seq_len, Q.device)
            Q = apply_rotary_pos_emb(Q, q_cos, q_sin)
            
            # Apply RoPE to K with its sequence length (if different from Q)
            if k_seq_len != q_seq_len:
                k_cos, k_sin = self.rope.get_embeddings(k_seq_len, K.device)
                K = apply_rotary_pos_emb(K, k_cos, k_sin)
            else:
                # Same sequence length, use same embeddings
                K = apply_rotary_pos_emb(K, q_cos, q_sin)
        
        relative_bias = None
        if self.pos_encoding == 'relative_bias':
            seq_len = Q.size(2)
            relative_bias = self.relative_bias(seq_len)

        # Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, relative_bias)
        
        # Combine heads back
        output = self._combine_heads(attention_output)
        
        # Apply output projection
        output = self.W_o(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """Feedforward network with two linear layers"""
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            ff_dim: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        
        # Two linear layers: d_model → ff_dim → d_model
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layer weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear layer with ReLU activation
        hidden = F.relu(self.linear1(x))
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Second linear layer
        output = self.linear2(hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output

class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention, cross-attention, and feedforward"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, pos_encoding: str = None, max_seq_len: int = 2048):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            dropout: Dropout rate
            pos_encoding: Type of positional encoding
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, pos_encoding, max_seq_len)
        
        # Cross-attention with encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout, pos_encoding, max_seq_len)
        
        # Feedforward network
        self.feedforward = FeedForward(d_model, ff_dim, dropout)
        
        # Layer normalization layers
        from utils import CustomLayerNorm
        self.norm1 = CustomLayerNorm(d_model)  # After self-attention
        self.norm2 = CustomLayerNorm(d_model)  # After cross-attention
        self.norm3 = CustomLayerNorm(d_model)  # After feedforward
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                self_attention_mask: Optional[torch.Tensor] = None,
                cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attention_mask: Mask for self-attention (look-ahead + padding)
            cross_attention_mask: Mask for cross-attention (padding only)
        
        Returns:
            Output tensor of same shape as input
        """
        # 1. Pre-LN: Masked self-attention sublayer
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(norm_x, norm_x, norm_x, self_attention_mask)
        x = x + self.dropout(attn_output)
        
        # 2. Pre-LN: Cross-attention sublayer
        norm_x = self.norm2(x)
        cross_attn_output, _ = self.cross_attention(norm_x, encoder_output, encoder_output, cross_attention_mask)
        x = x + self.dropout(cross_attn_output)
        
        # 3. Pre-LN: Feedforward sublayer
        norm_x = self.norm3(x)
        ff_output = self.feedforward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x

class Decoder(nn.Module):
    """Complete decoder with N decoder layers"""
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 ff_dim: int, max_seq_len: int = 2048, dropout: float = 0.1, 
                 pos_encoding: str = 'rope', pad_idx: int = 0):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pos_encoding: Positional encoding method ('rope' or 'relative_bias')
            pad_idx: Padding token index
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding = pos_encoding
        self.pad_idx = pad_idx
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_idx)
        
        # Positional encoding layer (for RoPE)
        self.pos_encoding_layer = None
        if pos_encoding == 'rope':
            # RoPE is applied in MultiHeadAttention, but we can keep a reference if needed
            pass
        elif pos_encoding == 'relative_bias':
            from utils import RelativePositionBias
            self.pos_encoding_layer = RelativePositionBias(max_seq_len, num_heads)
        else:
            # Sinusoidal or other positional encodings can be added here
            from utils import PositionalEncoding
            self.pos_encoding_layer = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout, pos_encoding, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        from utils import CustomLayerNorm
        self.final_norm = CustomLayerNorm(d_model)
        
        # Final linear projection to vocabulary size
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize output projection weights
        self._init_output_projection()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_output_projection(self):
        """Initialize output projection weights"""
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor] = None,
                cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attention_mask: Mask for self-attention (look-ahead + padding)
            cross_attention_mask: Mask for cross-attention (padding only)
        
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # 1. Token embeddings
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 2. Add positional encoding
        if self.pos_encoding_layer:
            x = self.pos_encoding_layer(x)
        
        # 3. Apply dropout
        x = self.dropout(x)
        
        # 4. Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
        
        # 5. Final layer normalization
        x = self.final_norm(x)
        
        # 6. Final linear projection to vocabulary size
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, encoder_output: torch.Tensor,
                              self_attention_mask: Optional[torch.Tensor] = None,
                              cross_attention_mask: Optional[torch.Tensor] = None) -> list:
        """
        Get attention weights from all layers for analysis
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attention_mask: Mask for self-attention
            cross_attention_mask: Mask for cross-attention
        
        Returns:
            List of attention weights from each layer
        """
        attention_weights = []
        
        # Get embeddings and positional encoding
        seq_len = x.size(1)
        x = self.token_embedding(x)
        
        if self.pos_encoding == 'rope':
            x = self.pos_encoding_layer(x, seq_len)
        
        x = self.dropout(x)
        
        # Pass through decoder layers and collect attention weights
        for layer in self.layers:
            # Get attention weights from the layer
            # Note: This requires modifying the layer to return attention weights
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
            # For now, we'll just pass through - attention weights can be added later
        
        x = self.final_norm(x)
        
        return attention_weights
