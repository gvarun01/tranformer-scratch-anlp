import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_and_plot_lengths(file_path, tokenizer, language_code):
    """Analyzes and plots sequence lengths for a given file."""
    print(f"Analyzing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Tokenize each line and get the length
    lengths = [len(tokenizer.encode(line.strip())) for line in lines if line.strip()]

    if not lengths:
        print(f"No content found in {file_path}.")
        return None

    # Calculate statistics
    avg_len = np.mean(lengths)
    max_len = np.max(lengths)
    percentile_98 = int(np.percentile(lengths, 98))

    print(f"  - Total lines processed: {len(lengths)}")
    print(f"  - Average sequence length: {avg_len:.2f} tokens")
    print(f"  - Maximum sequence length: {max_len} tokens")
    print(f"  - 98th percentile length: {percentile_98} tokens")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, alpha=0.8, color='skyblue', label=f'{language_code.upper()} Sequences')
    plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=2, label=f'Average: {avg_len:.2f}')
    plt.axvline(percentile_98, color='green', linestyle='dashed', linewidth=2, label=f'98th Percentile: {percentile_98}')
    plt.title(f'Sequence Length Distribution for {os.path.basename(file_path)}')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plot_filename = f'sequence_length_distribution_{language_code}.png'
    plt.savefig(plot_filename)
    print(f"  - Plot saved to {plot_filename}")
    plt.close()

    return percentile_98

def main():
    """Main function to run the analysis."""
    # --- Configuration ---
    en_tokenizer_path = 'models/spm_en.model'
    fi_tokenizer_path = 'models/spm_fi.model'
    en_file_path = 'data/EUbookshop.en'
    fi_file_path = 'data/EUbookshop.fi'
    # ---------------------

    print("Loading tokenizers...")
    # Load English tokenizer
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(en_tokenizer_path)

    # Load Finnish tokenizer
    sp_fi = spm.SentencePieceProcessor()
    sp_fi.load(fi_tokenizer_path)
    print("Tokenizers loaded successfully.")

    print("-" * 30)
    en_suggestion = analyze_and_plot_lengths(en_file_path, sp_en, 'en')
    print("-" * 30)
    fi_suggestion = analyze_and_plot_lengths(fi_file_path, sp_fi, 'fi')
    print("-" * 30)

    if en_suggestion and fi_suggestion:
        suggested_max_len = max(en_suggestion, fi_suggestion)
        print(f"\nRECOMMENDATION:")
        print(f"Based on the 98th percentile, a good value for 'max_sequence_length' would be {suggested_max_len}.")
        print("This covers the vast majority of sentences while optimizing memory and training speed.")

if __name__ == '__main__':
    main()
