#!/usr/bin/env python3
"""
Testing script for a trained Transformer model checkpoint.
"""

import torch
import argparse
import json
import sentencepiece as smp
from train import Transformer
from prepare_data import load_spm_model
from test import DecodingEngine, BLEUEvaluator

class SPMWrapper:
    """Wrapper for SentencePiece to mimic Vocabulary interface"""
    def __init__(self, sp_model):
        self.sp = sp_model
        self.vocab_size = len(sp_model)
    
    def __len__(self):
        return self.vocab_size
    
    def encode(self, text: str):
        if isinstance(text, str):
            return self.sp.encode_as_ids(text)
        else:
            # If it's already a list of tokens, join and encode
            return self.sp.encode_as_ids(' '.join(text))
    
    def decode(self, ids):
        return self.sp.decode_ids(ids)

def load_model_and_vocabs(model_path: str, config_path: str, src_vocab_path: str, 
                          tgt_vocab_path: str, device: torch.device):
    """Load a trained model and its vocabularies."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load checkpoint first to get the actual model dimensions
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get actual vocab sizes from the checkpoint's output projection layer
    decoder_weight_shape = checkpoint['model_state_dict']['decoder.output_projection.weight'].shape
    actual_tgt_vocab_size = decoder_weight_shape[0]  # Output vocab size
    
    # Get source vocab size from encoder embedding
    encoder_weight_shape = checkpoint['model_state_dict']['encoder.embedding.weight'].shape
    actual_src_vocab_size = encoder_weight_shape[0]  # Input vocab size
    
    print(f"Detected vocab sizes from checkpoint:")
    print(f"  Source vocab size: {actual_src_vocab_size}")
    print(f"  Target vocab size: {actual_tgt_vocab_size}")
    
    # Load SentencePiece models
    src_sp = load_spm_model(src_vocab_path)
    tgt_sp = load_spm_model(tgt_vocab_path)
    
    # Wrap them to mimic Vocabulary interface
    src_vocab = SPMWrapper(src_sp)
    tgt_vocab = SPMWrapper(tgt_sp)
    
    # Use actual vocabulary sizes from the checkpoint
    model = Transformer(
        src_vocab_size=actual_src_vocab_size,
        tgt_vocab_size=actual_tgt_vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pos_encoding=config['pos_encoding'],
        pad_idx=0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab

def main():
    parser = argparse.ArgumentParser(description='Test a Transformer checkpoint.')
    parser.add_argument('--checkpoint', type=str, default='models/rope/checkpoint_epoch_1_virat.pt', help='Path to the model checkpoint.')
    parser.add_argument('--config', type=str, default='models/rope/config.json', help='Path to the model config.')
    parser.add_argument('--src_vocab', type=str, default='models/spm_fi.model', help='Path to the source vocabulary.')
    parser.add_argument('--tgt_vocab', type=str, default='models/spm_en.model', help='Path to the target vocabulary.')
    parser.add_argument('--test_file', type=str, default='data/prepared/test.fi', help='Path to the test file (source language).')
    parser.add_argument('--ref_file', type=str, default='data/prepared/test.en', help='Path to the reference file (target language).')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., "cpu", "cuda").')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model, src_vocab, tgt_vocab = load_model_and_vocabs(args.checkpoint, args.config, args.src_vocab, args.tgt_vocab, device)
    
    print("✅ Model loaded successfully!")
    print(f"Model vocab sizes: src={model.src_vocab_size}, tgt={model.tgt_vocab_size}")
    
    engine = DecodingEngine(model, src_vocab, tgt_vocab, device)

    # --- Generate a few sample translations ---
    print("\n--- Sample Translations ---")
    sample_sentences = [
        "Tämä on lause.",
        "Mitä uutta?", 
        "Toivottavasti tämä käännös on hyvä.",
        "Hyvää päivää!",
        "Kiitos paljon."
    ]
    
    for sentence in sample_sentences:
        try:
            translation = engine.translate(sentence, strategy='greedy')
            print(f"Finnish: {sentence}")
            print(f"English: {translation}")
            print()
        except Exception as e:
            print(f"Error translating '{sentence}': {e}")
            print()

    print("Translation test completed!")

if __name__ == '__main__':
    main()
