# CSE 256 PA2: Transformer Blocks + Dissecting a Pretrained LLM

- **Part 1:** a Transformer encoder trained jointly with a feedforward classifier to predict which politician gave a speech segment.
- **Part 2:** a GPT-like word-level decoder pretrained on next-word prediction.
- **Part 3:** analysis of Pythia, in a separate notebook.

All attention and Transformer code in Parts 1 and 2 is written from scratch in `transformer.py`; no library Transformer layers are used.

## Requirements

- Python 3.9 or later
- PyTorch (a CPU is enough for Parts 1 and 2)
- NLTK, with the `punkt_tab` tokenizer data, which the provided tokenizer uses
- matplotlib

```bash
pip install torch nltk matplotlib
python -c "import nltk; nltk.download('punkt_tab')"
```

## Running Parts 1 and 2

Run the commands from this directory, which contains `speechesdataset/`.

```bash
python main.py part1   # Part 1: encoder + classifier
python main.py part2   # Part 2: decoder language model
python main.py         # both parts
```

Both parts use the default hyperparameters from the starter `main.py`:

- batch size 16, block size 32, learning rate 1e-3 (Adam)
- n_embd 64, 2 heads, 4 layers
- feedforward hidden size 100 inside every block
- classifier 64 → 100 → 3, trained for 15 epochs
- 500 language-model iterations, evaluated every 100

On a laptop CPU, Part 1 takes about 2 minutes of training and Part 2 about 40 seconds.

The output of `python main.py` from the reported run is saved in `logs/main_output.txt`. The extra runs used in the report (other seeds, `--no_pad_mask`, `--init default`) are saved next to it, in `logs/part1_*.txt` and `logs/part2_*.txt`.

### What each part prints and saves

| Part | Printed | Saved in `plots/` |
|---|---|---|
| 1 | Vocabulary size; encoder and classifier parameter counts; train loss, train accuracy and test accuracy after each of the 15 epochs; final test accuracy; per-head attention statistics on the test set; attention normalization checks | `part1_accuracy.png`; `part1_encoder_attention_s{1,2}.png` (all heads of all layers for two sentences); one map per head in `part1_encoder_attention_maps/` |
| 2 | Size and `<unk>` rate of each evaluation set; decoder parameter count; train and test perplexity (Obama, W. Bush, H. W. Bush) at iterations 0, 100, …, 500; per-head attention statistics; attention normalization checks | `part2_perplexity.png`; `part2_decoder_attention_s{1,2}.png`; one map per head in `part2_decoder_attention_maps/` |

### Options

| Option | Default | Description |
|---|---|---|
| `part1` / `part2` / `all` | `all` | Which part to run |
| `--seed` | `42` | Random seed. Runs are deterministic for a given seed on the same machine; different PyTorch builds or CPUs can give slightly different numbers |
| `--init` | `gpt2` | `gpt2`: N(0, 0.02) weights and zero biases. `default`: PyTorch's built-in initialization (used for the comparison in the report) |
| `--no_pad_mask` | off | Part 1 only: the encoder attends to padding and the mean includes padding positions (used for the comparison in the report) |
| `--plot_dir` | `plots` | Where figures are saved |

## Files

| File | Contents |
|---|---|
| `transformer.py` | Attention head, multi-head attention, feedforward layer, Transformer block (pre-LayerNorm), learned token + position embeddings, `TransformerEncoder`, `FeedForwardClassifier`, `EncoderClassifier` (encoder → mean over real tokens → classifier), `TransformerDecoder` (causal mask, LM head, cross-entropy loss) |
| `main.py` | Data loading, training loops for both parts, accuracy and perplexity evaluation, attention statistics, plots |
| `utilities.py` | Provided attention sanity check (rows sum to 1), plus a combined plot of all heads with token labels |
| `dataset.py`, `tokenizer.py` | Provided data classes and word-level tokenizer, with the small changes listed below |
| `PA2_part3.ipynb` | Part 3: logit lens and layer transplants on Pythia-160M. It runs on its own, ideally on a Colab GPU; see the notebook's setup cell |

## Changes to the starter code

- **`tokenizer.py`:** the vocabulary is sorted before ids are assigned. Python randomizes the iteration order of a set of strings from run to run, so without this, token ids, and therefore results, changed between runs even with a fixed seed.
- **`dataset.py`:** `LanguageModelingDataset` takes an optional `stride`. Training uses `stride=1`, the original behavior. Evaluation uses `stride=block_size`, so every token of an evaluation file is predicted exactly once.
- **`main.py`:**
  - `compute_perplexity` used an undefined `total_loss`. It now averages the loss over all predicted tokens.
  - `compute_classifier_accuracy` now unpacks the `(logits, attention maps)` tuple the models return.
  - `load_texts` sorts the file list so the run is reproducible.
- **`utilities.py`:**
  - The sanity check moves its input to the model's device.
  - It saves figures with a prefix into a folder instead of opening windows.
  - It prints when the normalization test passes.

## Acknowledgements

I used Claude (Anthropic) while working on this assignment.
