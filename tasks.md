
# 📝 Assignment 1 – Transformer From Scratch (Detailed Task Checklist)

## 1. Project Setup
* [X] **Setup the virtual env and make sure to use it everywhere**

* [X] **Create project structure**
  Create empty files:

  * `encoder.py` → encoder classes
  * `decoder.py` → decoder classes
  * `train.py` → training loop and main script
  * `test.py` → decoding strategies and evaluation
  * `utils.py` → helper functions (masks, attention, etc.)
  * `README.md` → instructions
  * `TASKS.md` → this checklist

* [X] **Setup configuration/argparse**
  Define hyperparameters:

  * `d_model`, `num_heads`, `num_layers`, `ff_dim`, `dropout`
  * `learning_rate`, `batch_size`, `num_epochs`, `max_len`
  * decoding strategy (`greedy`, `beam`, `topk`)
    Implement as either `config.json` or argparse in `train.py`.

---

## 2. Data Preprocessing
* [X] **Download dataset**
  Download Finnish–English parallel corpus. Save to `data/`.

* [X] **Data preprocessing pipeline (`utils.py`)**
  * Text cleaning and normalization
  * Handle special tokens (`<bos>`, `<eos>`, `<pad>`, `<unk>`)
  * Implement proper tokenization pipeline

* [ ] **Tokenization & vocabulary**

  * Train SentencePiece (or BPE) tokenizer for both Finnish (src) and English (tgt).
  * Build vocab dictionaries: `{token → id, id → token}`.
  * Save vocab files for reuse.

* [ ] **Train/val/test split**
  Split dataset into \~80/10/10 proportions.

* [X] **Implement Dataset class (`utils.py`)**

  * Create `TranslationDataset` subclass of `torch.utils.data.Dataset`.
  * Return `(src_tensor, tgt_tensor)` with `<bos>`, `<eos>`, `<pad>`.

* [X] **Implement DataLoader**

  * Use PyTorch DataLoader with custom collate\_fn.
  * Ensure dynamic padding to batch max length.

* [X] **Implement masks (`utils.py`)**

  * `create_padding_mask(seq, pad_idx)` → mask out pad tokens.
  * `create_look_ahead_mask(size)` → mask out pad tokens.

---

## 3. Core Transformer Components

* [X] **Token embeddings (`encoder.py`, `decoder.py`)**

  * Implement embedding layers for tokens.
  * Scale by `sqrt(d_model)`.

* [X] **Positional encodings – RoPE (`utils.py`)**

  * Implement rotary positional embeddings applied to Q/K vectors
  * Apply rotation matrices based on position
  * Handle different sequence lengths
  * Configurable via flag

* [X] **Positional encodings – Relative Position Bias (`utils.py`)**

  * Implement learned bias matrix added to attention scores
  * Handle relative distances between positions
  * Configurable via flag

* [X] **Scaled dot-product attention (`utils.py`)**

  * Input: Q, K, V, mask.
  * Compute: `softmax((QK^T)/sqrt(d_k) + mask) * V`.
  * Return (output, attention\_weights).

* [X] **Multi-head attention (`encoder.py`)**

  * Split input into `num_heads`.
  * Apply scaled dot-product attention per head.
  * Concatenate and project.

* [X] **Feedforward network (`encoder.py`)**

  * Two linear layers: `d_model → ff_dim → d_model`.
  * Activation: ReLU or GELU.

* [X] **LayerNorm (`utils.py`)**

  * Implement custom LayerNorm (no `torch.nn.LayerNorm`).
  * Learnable gamma, beta.

---

## 4. Encoder

* [X] **EncoderLayer (`encoder.py`)**

  * Multi-head self-attention + residual + norm.
  * Feedforward + residual + norm.

* [X] **Encoder (`encoder.py`)**

  * Stack N encoder layers.
  * Input: token embeddings + positional encodings.
  * Output: encoded sequence.

---

## 5. Decoder

* [X] **DecoderLayer (`decoder.py`)**

  * Masked multi-head self-attention + residual + norm.
  * Cross-attention with encoder output + residual + norm.
  * Feedforward + residual + norm.

* [X] **Decoder (`decoder.py`)**

  * Stack N decoder layers.
  * Input: token embeddings + positional encodings.
  * Final linear → vocab size logits.

---

## 6. Complete Model Architecture

* [X] **Complete Transformer Model (`train.py` or new `model.py`)**
  * Combine encoder and decoder
  * Handle source/target sequence flow
  * Implement forward pass logic

* [X] **Model initialization and configuration (`train.py`)**
  * Weight initialization strategy
  * Parameter counting
  * Model summary printing

---

## 7. Training Loop

* [X] **Loss function (`train.py`)**

  * Use `nn.CrossEntropyLoss(ignore_index=pad_idx)`.

* [X] **Optimizer & LR scheduler (`train.py`)**

  * Adam optimizer.
  * Learning rate warmup schedule (inverse sqrt decay).

* [X] **Training step (`train.py`)**

  * Forward pass with teacher forcing.
  * Compute loss.
  * Backprop + optimizer step.
  * Gradient clipping.

* [X] **Validation step (`train.py`)**

  * Run on validation set.
  * Compute average loss.

* [X] **Training monitoring (`train.py`)**
  * Learning rate tracking
  * Gradient norm monitoring
  * Parameter statistics tracking
  * Early stopping mechanism

* [X] **Checkpoint saving (`train.py`)**

  * Save model + optimizer every epoch.

* [X] **Logging (`train.py`)**

  * Track training and validation loss.
  * Save to file.

---

## 8. Decoding Strategies

* [X] **Greedy decoding (`test.py`)**

  * Select `argmax` at each step until `<eos>` or max length.

* [X] **Beam search decoding (`test.py`)**

  * Keep top-B beams.
  * Expand and prune at each step.
  * Return best sequence.

* [X] **Top-k sampling decoding (`test.py`)**

  * At each step, sample from top-k tokens (e.g., k=50).

* [X] **Configurable decoding strategy (`test.py`)**

  * Add CLI flag `--decoding_strategy [greedy|beam|topk]`.

---

## 9. Evaluation

* [X] **BLEU computation (`test.py`)**

  * Use sacrebleu to compute BLEU on test translations.

* [ ] **Compare decoding strategies (`report.pdf`)**

  * Report BLEU for greedy, beam, and top-k.

* [ ] **Training curves (`report.pdf`)**

  * Plot loss vs epochs for RoPE and Relative Bias.

---

## 10. Testing and Validation

* [ ] **Unit tests for core components (`test_components.py`)**
  * Test attention mechanism
  * Test positional encodings
  * Test forward/backward passes

---

## 11. Report & Packaging

* [ ] **Write report (`report.pdf`)**
  Include:

  * BLEU comparison table.
  * Training curves.
  * Observations (translation quality + convergence).

* [ ] **README.md**

  * Environment setup.
  * How to train.
  * How to test.
  * Pretrained model link.
