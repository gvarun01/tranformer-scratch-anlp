#!/usr/bin/env python3
"""
Test script for decoding strategies implementation
"""

import torch
import torch.nn as nn
from test import DecodingEngine, BLEUEvaluator
from train import Transformer
from utils import Vocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN

def create_dummy_vocab(size: int = 1000) -> Vocabulary:
    """Create a dummy vocabulary for testing"""
    vocab = Vocabulary(min_freq=1)
    
    # Add special tokens first
    for token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        vocab.token2id[token] = len(vocab.token2id)
        vocab.id2token[len(vocab.id2token)] = token
    
    # Add dummy tokens
    for i in range(4, size):
        token = f"word_{i}"
        vocab.token2id[token] = i
        vocab.id2token[i] = token
    
    return vocab

def create_dummy_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    """Create a dummy transformer model for testing"""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        max_seq_len=512,
        dropout=0.1,
        pos_encoding='rope',
        pad_idx=0
    )
    return model

def test_decoding_engines():
    """Test all decoding strategies"""
    print("Testing decoding strategies...")
    
    # Create dummy vocabularies
    src_vocab = create_dummy_vocab(1000)
    tgt_vocab = create_dummy_vocab(800)
    
    # Create dummy model
    model = create_dummy_model(len(src_vocab), len(tgt_vocab))
    
    # Set device
    device = torch.device('cpu')
    model.to(device)
    
    # Create decoding engine
    engine = DecodingEngine(model, src_vocab, tgt_vocab, device, max_len=20)
    
    # Test source text
    test_text = "Hello world this is a test"
    
    print(f"Testing with input: '{test_text}'")
    print("-" * 50)
    
    # Test greedy decoding
    try:
        print("1. Testing Greedy Decoding...")
        translation, score = engine.greedy_decode(test_text)
        print(f"   ✅ Greedy: '{translation}' (score: {score:.4f})")
    except Exception as e:
        print(f"   ❌ Greedy failed: {e}")
    
    # Test beam search
    try:
        print("2. Testing Beam Search...")
        translation, score = engine.beam_search_decode(test_text, beam_size=3)
        print(f"   ✅ Beam: '{translation}' (score: {score:.4f})")
    except Exception as e:
        print(f"   ❌ Beam search failed: {e}")
    
    # Test top-k sampling
    try:
        print("3. Testing Top-k Sampling...")
        translation, score = engine.top_k_sampling_decode(test_text, k=10)
        print(f"   ✅ Top-k: '{translation}' (score: {score:.4f})")
    except Exception as e:
        print(f"   ❌ Top-k sampling failed: {e}")

def test_bleu_evaluator():
    """Test BLEU evaluation"""
    print("\nTesting BLEU evaluator...")
    
    evaluator = BLEUEvaluator()
    
    # Test data
    predictions = [
        "Hello world",
        "This is a test sentence",
        "Machine translation is working"
    ]
    
    references = [
        "Hello world",
        "This is a test sentence",
        "Machine translation works well"
    ]
    
    try:
        results = evaluator.compute_bleu(predictions, references)
        print(f"✅ BLEU computation successful:")
        print(f"   BLEU Score: {results['bleu']:.2f}")
        print(f"   Brevity Penalty: {results['bp']:.3f}")
        print(f"   Length Ratio: {results['ratio']:.3f}")
    except Exception as e:
        print(f"❌ BLEU evaluation failed: {e}")

def test_vocabulary_ops():
    """Test vocabulary operations"""
    print("\nTesting vocabulary operations...")
    
    vocab = create_dummy_vocab(100)
    
    # Test encoding
    tokens = ["Hello", "world", UNK_TOKEN]
    try:
        encoded = vocab.encode(tokens)
        print(f"✅ Encoding: {tokens} -> {encoded}")
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
    
    # Test decoding
    try:
        decoded = vocab.decode(encoded)
        print(f"✅ Decoding: {encoded} -> {decoded}")
    except Exception as e:
        print(f"❌ Decoding failed: {e}")

def main():
    """Main test function"""
    print("🧪 Testing Decoding Implementation")
    print("=" * 60)
    
    # Test vocabulary operations
    test_vocabulary_ops()
    
    # Test BLEU evaluator
    test_bleu_evaluator()
    
    # Test decoding engines
    test_decoding_engines()
    
    print("\n" + "=" * 60)
    print("✨ All tests completed!")

if __name__ == "__main__":
    main()
