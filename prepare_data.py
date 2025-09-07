#!/usr/bin/env python3
"""
Data preparation script for Transformer from scratch.
This script trains SentencePiece tokenizers and splits the data into train, validation, and test sets.
"""

import os
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import argparse

def train_sentencepiece_model(input_file: str, model_prefix: str, vocab_size: int, lang: str):
    """
    Train a SentencePiece model.
    
    Args:
        input_file: Path to the input text file.
        model_prefix: Prefix for the model and vocab files.
        vocab_size: Size of the vocabulary.
        lang: Language identifier (e.g., 'en', 'fi').
    """
    print(f"Training SentencePiece model for {lang}...")
    
    # Define command for SentencePiece training
    command = (
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage=1.0 '
        f'--model_type=bpe '
        f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        f'--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<bos> --eos_piece=<eos>'
    )
    
    # Train the model
    spm.SentencePieceTrainer.Train(command)
    
    print(f"✅ {lang} model trained and saved as {model_prefix}.model")

def split_data(src_file: str, tgt_file: str, output_dir: str, test_size: float = 0.1, val_size: float = 0.1):
    """
    Split parallel data into train, validation, and test sets.
    
    Args:
        src_file: Path to the source language file.
        tgt_file: Path to the target language file.
        output_dir: Directory to save the split files.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the dataset to include in the validation split.
    """
    print("Splitting data into train, validation, and test sets...")
    
    # Read source and target files
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    
    assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines."
    
    # Split into train+val and test
    src_train_val, src_test, tgt_train_val, tgt_test = train_test_split(
        src_lines, tgt_lines, test_size=test_size, random_state=42
    )
    
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src_train_val, tgt_train_val, test_size=val_size_adjusted, random_state=42
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to save split files
    def save_split(lines: list, filename: str):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"  Saved {len(lines)} lines to {path}")
    
    # Save all splits
    save_split(src_train, 'train.fi')
    save_split(tgt_train, 'train.en')
    save_split(src_val, 'val.fi')
    save_split(tgt_val, 'val.en')
    save_split(src_test, 'test.fi')
    save_split(tgt_test, 'test.en')
    
    print("✅ Data splitting complete.")

def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(description='Data Preparation for Transformer')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory with raw data')
    parser.add_argument('--output_dir', type=str, default='data/prepared', help='Directory for processed data')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save tokenizer models')
    
    # Tokenizer settings
    parser.add_argument('--src_vocab_size', type=int, default=8000, help='Source vocabulary size')
    parser.add_argument('--tgt_vocab_size', type=int, default=8000, help='Target vocabulary size')
    
    # Data split settings
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # File paths
    src_file = os.path.join(args.data_dir, 'EUbookshop.fi')
    tgt_file = os.path.join(args.data_dir, 'EUbookshop.en')
    
    # Train SentencePiece models
    train_sentencepiece_model(
        input_file=src_file,
        model_prefix=os.path.join(args.model_dir, 'spm_fi'),
        vocab_size=args.src_vocab_size,
        lang='fi'
    )
    
    train_sentencepiece_model(
        input_file=tgt_file,
        model_prefix=os.path.join(args.model_dir, 'spm_en'),
        vocab_size=args.tgt_vocab_size,
        lang='en'
    )
    
    # Split data
    split_data(
        src_file=src_file,
        tgt_file=tgt_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    print("\nData preparation complete!")
    print(f"Tokenizer models saved in: {args.model_dir}")
    print(f"Split data saved in: {args.output_dir}")

def load_spm_model(model_path: str):
    """
    Load a SentencePiece model.
    
    Args:
        model_path: Path to the SentencePiece model file (.model).
        
    Returns:
        SentencePiece processor.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

if __name__ == "__main__":
    main()
