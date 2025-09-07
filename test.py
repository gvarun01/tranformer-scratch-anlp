#!/usr/bin/env python3
"""
Decoding strategies and evaluation for Transformer from scratch
This file contains inference implementations: greedy, beam search, top-k sampling, and BLEU evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import time
from typing import List, Tuple, Optional, Dict
from sacrebleu import BLEU
from train import Transformer
from utils import (
    create_padding_mask, 
    create_combined_mask, 
    Vocabulary, 
    text_preprocessor,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    greedy_decode,
    beam_search_decode,
    top_k_sampling_decode
)

class DecodingEngine:
    """Engine for different decoding strategies"""
    
    def __init__(self, model: Transformer, src_vocab: Vocabulary, tgt_vocab: Vocabulary, 
                 device: torch.device, max_len: int = 100):
        """
        Args:
            model: Trained transformer model
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            device: Device to run inference on
            max_len: Maximum generation length
        """
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.max_len = max_len
        
        # Get special token IDs
        self.pad_id = tgt_vocab.token2id[PAD_TOKEN]
        self.bos_id = tgt_vocab.token2id[BOS_TOKEN]
        self.eos_id = tgt_vocab.token2id[EOS_TOKEN]
        self.unk_id = tgt_vocab.token2id[UNK_TOKEN]
        
        # Set model to evaluation mode
        self.model.eval()
    
    def preprocess_source(self, src_text: str) -> torch.Tensor:
        """Preprocess source text and convert to tensor"""
        # Clean the text but don't add special tokens here
        cleaned_text = text_preprocessor.preprocess_text(src_text, add_special_tokens=False)
        
        # Encode with SentencePiece (which will add BOS/EOS internally)
        token_ids = self.src_vocab.encode(cleaned_text)
        
        # Convert to tensor and add batch dimension
        src_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        return src_tensor
    
    def postprocess_target(self, token_ids: List[int]) -> str:
        """Convert target token IDs back to text"""
        # Remove special tokens
        filtered_ids = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id not in [self.pad_id, self.bos_id]:
                filtered_ids.append(token_id)
        
        # Convert IDs to text using SentencePiece
        text = self.tgt_vocab.decode(filtered_ids)
        
        return text
    
    def translate(self, src_text: str, strategy: str = 'greedy', beam_width: int = 3, k: int = 5) -> str:
        """
        Translate source text using the specified decoding strategy.
        
        Args:
            src_text: Source text to translate.
            strategy: Decoding strategy ('greedy', 'beam', 'top_k').
            beam_width: Beam width for beam search.
            k: Value of k for top-k sampling.
            
        Returns:
            Translated text.
        """
        src_tensor = self.preprocess_source(src_text)
        src_mask = create_padding_mask(src_tensor, self.pad_id)

        if strategy == 'greedy':
            output_tokens = greedy_decode(self.model, src_tensor, src_mask, self.max_len, self.bos_id, self.eos_id, self.device)
        elif strategy == 'beam':
            output_tokens = beam_search_decode(self.model, src_tensor, src_mask, self.max_len, self.bos_id, self.eos_id, self.device, beam_width)
        elif strategy == 'top_k':
            output_tokens = top_k_sampling_decode(self.model, src_tensor, src_mask, self.max_len, self.bos_id, self.eos_id, self.device, k)
        else:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")
            
        return self.postprocess_target(output_tokens.squeeze().tolist())
    
    def beam_search_decode(self, src_text: str, beam_size: int = 5) -> Tuple[str, float]:
        """
        Beam search decoding: keep top-B beams and expand
        
        Args:
            src_text: Source text string
            beam_size: Number of beams to keep
        
        Returns:
            Tuple of (best_decoded_text, best_score)
        """
        with torch.no_grad():
            # Preprocess source
            src_tensor = self.preprocess_source(src_text)
            
            # Encode source
            encoder_output = self.model.encode(src_tensor)
            
            # Initialize beams: (sequence, log_prob, finished)
            beams = [([self.bos_id], 0.0, False)]
            
            for step in range(self.max_len):
                new_beams = []
                
                for sequence, log_prob, finished in beams:
                    if finished:
                        new_beams.append((sequence, log_prob, finished))
                        continue
                    
                    # Create tensor for current sequence
                    tgt_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                    tgt_mask = create_combined_mask(tgt_tensor, self.pad_id)
                    
                    # Decode
                    logits = self.model.decode(tgt_tensor, encoder_output, tgt_mask)
                    
                    # Get logits for the last position
                    next_token_logits = logits[0, -1, :]  # (vocab_size,)
                    
                    # Apply log softmax
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top-k candidates
                    top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size)
                    
                    # Expand beam
                    for i in range(beam_size):
                        next_token_id = top_k_indices[i].item()
                        next_log_prob = top_k_log_probs[i].item()
                        
                        new_sequence = sequence + [next_token_id]
                        new_log_prob = log_prob + next_log_prob
                        
                        # Check if sequence is finished
                        is_finished = (next_token_id == self.eos_id)
                        
                        new_beams.append((new_sequence, new_log_prob, is_finished))
                
                # Keep top beam_size beams
                new_beams.sort(key=lambda x: x[1] / len(x[0]), reverse=True)  # Length normalized
                beams = new_beams[:beam_size]
                
                # Check if all beams are finished
                if all(finished for _, _, finished in beams):
                    break
            
            # Select best beam
            best_sequence, best_log_prob, _ = max(beams, key=lambda x: x[1] / len(x[0]))
            
            # Convert to text
            decoded_text = self.postprocess_target(best_sequence)
            
            # Calculate average log probability
            avg_log_prob = best_log_prob / len(best_sequence)
            
            return decoded_text, avg_log_prob
    
    def top_k_sampling_decode(self, src_text: str, k: int = 50, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Top-k sampling decoding: sample from top-k tokens
        
        Args:
            src_text: Source text string
            k: Number of top tokens to consider
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Tuple of (decoded_text, average_log_prob)
        """
        with torch.no_grad():
            # Preprocess source
            src_tensor = self.preprocess_source(src_text)
            
            # Encode source
            encoder_output = self.model.encode(src_tensor)
            
            # Initialize target sequence with BOS token
            tgt_sequence = [self.bos_id]
            tgt_tensor = torch.tensor([tgt_sequence], dtype=torch.long).to(self.device)
            
            log_probs = []
            
            for step in range(self.max_len):
                # Create target mask
                tgt_mask = create_combined_mask(tgt_tensor, self.pad_id)
                
                # Decode
                logits = self.model.decode(tgt_tensor, encoder_output, tgt_mask)
                
                # Get logits for the last position
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Get top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                
                # Apply softmax to top-k logits
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from top-k distribution
                sampled_idx = torch.multinomial(top_k_probs, 1).item()
                next_token_id = top_k_indices[sampled_idx].item()
                
                # Calculate log probability for the sampled token
                all_probs = F.softmax(next_token_logits, dim=-1)
                log_prob = torch.log(all_probs[top_k_indices[sampled_idx]]).item()
                log_probs.append(log_prob)
                
                # Add to sequence
                tgt_sequence.append(next_token_id)
                
                # Update tensor
                tgt_tensor = torch.tensor([tgt_sequence], dtype=torch.long).to(self.device)
                
                # Check if EOS token is generated
                if next_token_id == self.eos_id:
                    break
            
            # Calculate average log probability
            avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0.0
            
            # Convert to text
            decoded_text = self.postprocess_target(tgt_sequence)
            
            return decoded_text, avg_log_prob

class BLEUEvaluator:
    """BLEU score evaluation using sacrebleu"""
    
    def __init__(self):
        self.bleu = BLEU()
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores
        
        Args:
            predictions: List of predicted sentences
            references: List of reference sentences
        
        Returns:
            Dictionary with BLEU scores and statistics
        """
        # Compute BLEU score
        bleu_score = self.bleu.corpus_score(predictions, [references])
        
        return {
            'bleu': bleu_score.score,
            'bp': bleu_score.bp,  # Brevity penalty
            'ratio': bleu_score.ratio,
            'hyp_len': bleu_score.sys_len,
            'ref_len': bleu_score.ref_len,
            'precisions': bleu_score.precisions
        }
    
    def evaluate_file(self, pred_file: str, ref_file: str) -> Dict[str, float]:
        """
        Evaluate BLEU score from files
        
        Args:
            pred_file: Path to predictions file
            ref_file: Path to references file
        
        Returns:
            Dictionary with BLEU scores and statistics
        """
        # Read predictions
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f]
        
        # Read references
        with open(ref_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
        
        return self.compute_bleu(predictions, references)

def load_model_and_vocabs(model_path: str, config_path: str, src_vocab_path: str, 
                          tgt_vocab_path: str, device: torch.device) -> Tuple[Transformer, Vocabulary, Vocabulary]:
    """
    Load trained model and vocabularies
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to model configuration
        src_vocab_path: Path to source vocabulary
        tgt_vocab_path: Path to target vocabulary
        device: Device to load model on
    
    Returns:
        Tuple of (model, src_vocab, tgt_vocab)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabularies
    src_vocab = Vocabulary.load(src_vocab_path)
    tgt_vocab = Vocabulary.load(tgt_vocab_path)
    
    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pos_encoding=config['pos_encoding'],
        pad_idx=0
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab

def translate_text(engine: DecodingEngine, text: str, strategy: str, **kwargs) -> Tuple[str, float]:
    """
    Translate a single text using specified strategy
    
    Args:
        engine: DecodingEngine instance
        text: Source text to translate
        strategy: Decoding strategy ('greedy', 'beam', 'topk')
        **kwargs: Additional arguments for specific strategies
    
    Returns:
        Tuple of (translated_text, score)
    """
    if strategy == 'greedy':
        return engine.greedy_decode(text)
    elif strategy == 'beam':
        beam_size = kwargs.get('beam_size', 5)
        return engine.beam_search_decode(text, beam_size=beam_size)
    elif strategy == 'topk':
        k = kwargs.get('k', 50)
        temperature = kwargs.get('temperature', 1.0)
        return engine.top_k_sampling_decode(text, k=k, temperature=temperature)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")

def translate_file(engine: DecodingEngine, input_file: str, output_file: str, 
                   strategy: str, **kwargs) -> None:
    """
    Translate sentences from input file and save to output file
    
    Args:
        engine: DecodingEngine instance
        input_file: Path to input file with source sentences
        output_file: Path to output file for translations
        strategy: Decoding strategy
        **kwargs: Additional arguments for specific strategies
    """
    print(f"Translating {input_file} using {strategy} decoding...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f if line.strip()]
    
    translations = []
    scores = []
    
    start_time = time.time()
    
    for i, src_text in enumerate(source_sentences):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i + 1}/{len(source_sentences)} sentences "
                  f"({elapsed:.1f}s, {(i + 1) / elapsed:.1f} sent/s)")
        
        try:
            translation, score = translate_text(engine, src_text, strategy, **kwargs)
            translations.append(translation)
            scores.append(score)
        except Exception as e:
            print(f"  Error translating sentence {i + 1}: {e}")
            translations.append("")
            scores.append(0.0)
    
    # Save translations
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
    
    # Print statistics
    avg_score = sum(scores) / len(scores) if scores else 0.0
    total_time = time.time() - start_time
    
    print(f"  Completed: {len(translations)} sentences")
    print(f"  Average score: {avg_score:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Speed: {len(translations) / total_time:.1f} sent/s")
    print(f"  Output saved to: {output_file}")

def compare_strategies(engine: DecodingEngine, test_sentences: List[str], 
                       reference_sentences: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare different decoding strategies on test sentences
    
    Args:
        engine: DecodingEngine instance
        test_sentences: List of source sentences
        reference_sentences: List of reference translations
    
    Returns:
        Dictionary with results for each strategy
    """
    evaluator = BLEUEvaluator()
    strategies = ['greedy', 'beam', 'topk']
    results = {}
    
    for strategy in strategies:
        print(f"\nEvaluating {strategy} decoding...")
        
        predictions = []
        scores = []
        start_time = time.time()
        
        # Generate translations
        for i, src_text in enumerate(test_sentences):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_sentences)}")
            
            if strategy == 'greedy':
                pred, score = engine.greedy_decode(src_text)
            elif strategy == 'beam':
                pred, score = engine.beam_search_decode(src_text, beam_size=5)
            elif strategy == 'topk':
                pred, score = engine.top_k_sampling_decode(src_text, k=50)
            
            predictions.append(pred)
            scores.append(score)
        
        # Compute BLEU score
        bleu_results = evaluator.compute_bleu(predictions, reference_sentences)
        
        # Store results
        results[strategy] = {
            'bleu': bleu_results['bleu'],
            'avg_score': sum(scores) / len(scores),
            'time': time.time() - start_time,
            'speed': len(test_sentences) / (time.time() - start_time)
        }
        
        print(f"  BLEU: {bleu_results['bleu']:.2f}")
        print(f"  Avg Score: {results[strategy]['avg_score']:.4f}")
        print(f"  Time: {results[strategy]['time']:.1f}s")
    
    return results

def interactive_translation(engine: DecodingEngine, strategy: str = 'greedy') -> None:
    """
    Interactive translation mode
    
    Args:
        engine: DecodingEngine instance
        strategy: Default decoding strategy
    """
    print(f"\n🤖 Interactive Translation Mode (Strategy: {strategy})")
    print("Enter source text to translate (or 'quit' to exit):")
    print("Commands: 'strategy <name>' to change strategy, 'help' for help")
    print("-" * 60)
    
    current_strategy = strategy
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  strategy greedy/beam/topk - Change decoding strategy")
                print("  quit/exit/q - Exit interactive mode")
                print("  <text> - Translate the text")
                continue
            elif user_input.lower().startswith('strategy '):
                new_strategy = user_input.split()[1].lower()
                if new_strategy in ['greedy', 'beam', 'topk']:
                    current_strategy = new_strategy
                    print(f"Strategy changed to: {current_strategy}")
                else:
                    print("Invalid strategy. Use: greedy, beam, or topk")
                continue
            elif not user_input:
                continue
            
            # Translate
            print(f"Translating with {current_strategy} decoding...")
            start_time = time.time()
            
            translation, score = translate_text(engine, user_input, current_strategy)
            
            elapsed = time.time() - start_time
            
            print(f"Translation: {translation}")
            print(f"Score: {score:.4f}, Time: {elapsed:.2f}s")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transformer Translation Testing')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--src_vocab', type=str, required=True,
                       help='Path to source vocabulary file')
    parser.add_argument('--tgt_vocab', type=str, required=True,
                       help='Path to target vocabulary file')
    
    # Decoding strategy
    parser.add_argument('--decoding_strategy', type=str, 
                       choices=['greedy', 'beam', 'topk'], default='greedy',
                       help='Decoding strategy to use')
    
    # Strategy-specific parameters
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size for beam search')
    parser.add_argument('--k', type=int, default=50,
                       help='k value for top-k sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for top-k sampling')
    
    # Input/output
    parser.add_argument('--input_file', type=str,
                       help='Input file with source sentences')
    parser.add_argument('--output_file', type=str,
                       help='Output file for translations')
    parser.add_argument('--reference_file', type=str,
                       help='Reference file for BLEU evaluation')
    
    # Operation mode
    parser.add_argument('--mode', type=str, 
                       choices=['interactive', 'translate', 'evaluate', 'compare'],
                       default='interactive',
                       help='Operation mode')
    
    # Other parameters
    parser.add_argument('--max_len', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = get_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and vocabularies
    print("Loading model and vocabularies...")
    try:
        model, src_vocab, tgt_vocab = load_model_and_vocabs(
            args.model_path, args.config_path, args.src_vocab, args.tgt_vocab, device
        )
        print(f"✅ Model loaded successfully")
        print(f"   Source vocab size: {len(src_vocab):,}")
        print(f"   Target vocab size: {len(tgt_vocab):,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create decoding engine
    engine = DecodingEngine(model, src_vocab, tgt_vocab, device, args.max_len)
    
    # Execute based on mode
    if args.mode == 'interactive':
        interactive_translation(engine, args.decoding_strategy)
    
    elif args.mode == 'translate':
        if not args.input_file or not args.output_file:
            print("❌ Error: --input_file and --output_file required for translate mode")
            return
        
        print(f"Translating {args.input_file} to {args.output_file} using {args.decoding_strategy}...")
        
        with open(args.input_file, 'r', encoding='utf-8') as f_in, open(args.output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                src_text = line.strip()
                if src_text:
                    translated_text = engine.translate(
                        src_text, 
                        strategy=args.decoding_strategy, 
                        beam_width=args.beam_size, 
                        k=args.k
                    )
                    f_out.write(translated_text + '\n')
        
        print("Translation complete.")

    elif args.mode == 'evaluate':
        if not args.output_file or not args.reference_file:
            print("❌ Error: --output_file and --reference_file required for evaluate mode")
            return
        
        evaluator = BLEUEvaluator()
        results = evaluator.evaluate_file(args.output_file, args.reference_file)
        
        print(f"\n📊 BLEU Evaluation Results:")
        print(f"   BLEU Score: {results['bleu']:.2f}")
        print(f"   Brevity Penalty: {results['bp']:.3f}")
        print(f"   Length Ratio: {results['ratio']:.3f}")
        print(f"   Hypothesis Length: {results['hyp_len']}")
        print(f"   Reference Length: {results['ref_len']}")
        print(f"   Precisions: {[f'{p:.1f}' for p in results['precisions']]}")
    
    elif args.mode == 'compare':
        if not args.input_file or not args.reference_file:
            print("❌ Error: --input_file and --reference_file required for compare mode")
            return
        
        # Load test data
        with open(args.input_file, 'r', encoding='utf-8') as f:
            test_sentences = [line.strip() for line in f if line.strip()]
        
        with open(args.reference_file, 'r', encoding='utf-8') as f:
            reference_sentences = [line.strip() for line in f if line.strip()]
        
        # Limit to first 100 sentences for quick comparison
        if len(test_sentences) > 100:
            print(f"Limiting comparison to first 100 sentences (out of {len(test_sentences)})")
            test_sentences = test_sentences[:100]
            reference_sentences = reference_sentences[:100]
        
        # Compare strategies
        results = compare_strategies(engine, test_sentences, reference_sentences)
        
        # Print comparison table
        print(f"\n📊 Strategy Comparison Results:")
        print(f"{'Strategy':<10} {'BLEU':<8} {'Avg Score':<10} {'Time':<8} {'Speed':<10}")
        print("-" * 50)
        for strategy, metrics in results.items():
            print(f"{strategy:<10} {metrics['bleu']:<8.2f} {metrics['avg_score']:<10.4f} "
                  f"{metrics['time']:<8.1f} {metrics['speed']:<10.1f}")

if __name__ == "__main__":
    main()
