# CSE 256 PA2: Transformer Blocks + Dissecting a Pretrained LLM

Garvit Gopalani · CSE 256, UC San Diego, Fall 2026

- **Part 1:** a Transformer encoder trained jointly with a feedforward classifier, to predict which politician gave a speech segment.
- **Part 2:** a GPT-like word-level decoder pretrained on next-word prediction.
- **Part 3:** an analysis of Pythia-160M checkpoints in a Jupyter notebook.

## How to run

### 1. Set up the environment (once)

```bash
conda create -n cse256 python=3.11
conda activate cse256
pip install torch nltk matplotlib numpy
python -c "import nltk; nltk.download('punkt_tab')"
```

The `punkt_tab` download is needed by the provided word-level tokenizer. Parts 1 and 2 run on a CPU; a GPU is used automatically if one is available.

### 2. Parts 1 and 2

Run from the `PA2_code` folder, which contains `speechesdataset/`:

```bash
cd PA2_code
python main.py part1   # Part 1: encoder + classifier (about 2 min on a laptop CPU)
python main.py part2   # Part 2: decoder language model (about 40 s)
python main.py         # both parts
```

Both parts use the default hyperparameters from the starter `main.py`:

| Setting | Value |
|---|---|
| Batch size | 16 |
| Block size | 32 |
| Optimizer | Adam, learning rate 1e-3 |
| Model size | n_embd 64, 2 heads, 4 layers |
| Feedforward hidden size in every block | 100 |
| Classifier | 64 → 100 → 3, trained for 15 epochs |
| Language model | 500 iterations, evaluated every 100 |

To save the printed output to a file:

```bash
python main.py > logs/main_output.txt
```

On Windows, use `logs\main_output.txt`.

**Output.**
- **Part 1** prints:
  - the vocabulary size and parameter counts;
  - train loss, train accuracy and test accuracy after each of the 15 epochs, and the final test accuracy;
  - per-head attention statistics;
  - the check that every attention row sums to 1.
- **Part 2** prints:
  - the decoder parameter count;
  - train and test perplexity (Obama, W. Bush, H. W. Bush) at iterations 0, 100, …, 500;
  - the test loss on unseen words;
  - per-head attention statistics;
  - the attention row-sum check.
- **Plots** are saved in `PA2_code/plots/`.

### 3. Extra runs used in the report (optional)

```bash
python main.py part1 --seed 1 --plot_dir plots_extra          # also seeds 2 and 3
python main.py part1 --no_pad_mask --plot_dir plots_extra     # encoder attends to padding
python main.py part1 --init default --plot_dir plots_extra    # PyTorch's default initialization
python main.py part2 --seed 1 --plot_dir plots_extra          # also seeds 2 and 3
python main.py part2 --init default --plot_dir plots_extra
python data_stats.py                                          # data statistics for the Part 2 discussion
```

The outputs of all reported runs are saved in `PA2_code/logs/`.

| Option | Default | Description |
|---|---|---|
| `part1` / `part2` / `all` | `all` | Which part to run |
| `--seed` | `42` | Random seed. Results are the same for a given seed on the same machine; different PyTorch builds or CPUs can give slightly different numbers |
| `--init` | `gpt2` | `gpt2`: N(0, 0.02) weights. `default`: PyTorch's built-in initialization |
| `--no_pad_mask` | off | Part 1 only: do not mask padding in attention and in the mean |
| `--plot_dir` | `plots` | Where figures are saved |

### 4. Part 3 (notebook)

`PA2_code/PA2_part3.ipynb` runs on its own; Colab is recommended.

1. Open the notebook in Colab and choose *Runtime → Change runtime type → T4 GPU*.
2. Upload `PA2_code/speechesdataset/test_LM_obama.txt` to `/content`, Colab's working folder.
3. Choose *Runtime → Run all*. The notebook downloads six Pythia-160M checkpoints from Hugging Face; the submitted run took 3.2 minutes on a T4. It uses `torch` and `transformers`, which Colab already has.

The executed notebook, with all outputs, is in this repository. Its PDF export is `PA2_code/PA2_part3.pdf`.

## Results (from `PA2_code/logs/`)

| | Result |
|---|---|
| Part 1: final test accuracy (seed 42) | **87.33%** (best 87.60% at epoch 9); 87.5 ± 1.4% over 4 seeds |
| Part 1: encoder parameters | 481,424 (classifier 6,803) |
| Part 2: final perplexity, train / Obama / W. Bush / H. W. Bush | **68.3 / 319.0 / 474.1 / 365.9** |
| Part 2: decoder parameters | 847,634 |

The full results and discussion are in `report/PA2_final_report.pdf`. Part A covers Parts 1–2; Part B is the executed Part 3 notebook.

## Repository layout

```
CSE256_PA2_FA26/
├── README.md                  this file
├── CSE256_PA2_FA26.pdf        assignment handout
├── PA2_code/
│   ├── main.py                data loading, training loops for Parts 1 and 2, evaluation, plots
│   ├── transformer.py         encoder, classifier and decoder, written from scratch
│   ├── utilities.py           attention sanity checks and attention-map plots
│   ├── dataset.py             data classes (provided; added a stride option for evaluation)
│   ├── tokenizer.py           word-level tokenizer (provided; vocabulary sorted for reproducibility)
│   ├── data_stats.py          data statistics behind the Part 2 discussion
│   ├── PA2_part3.ipynb        Part 3 notebook, executed
│   ├── PA2_part3.pdf          PDF export of the notebook
│   ├── README.md              code-level details
│   ├── speechesdataset/       data
│   ├── logs/                  outputs of every reported run
│   ├── plots/                 figures from the main run
│   └── plots_extra/           figures from the extra runs
└── report/
    ├── PA2_final_report.pdf   Part A (Parts 1–2) + Part B (Part 3 notebook), submitted to Gradescope
    └── figures/               attention-map figures used in the report
```

## Implementation notes

- All attention and Transformer code for Parts 1 and 2 is written from scratch in `transformer.py`. No library Transformer layers are used. The model uses learned absolute position embeddings and pre-LayerNorm blocks.
- **Encoder:** padding is masked in attention, and the classifier takes the mean of the encoder outputs over the real (non-padding) tokens.
- **Decoder:** uses a causal mask and returns the mean cross-entropy over all positions.
- **Changes to the starter code:**
  - the vocabulary is sorted, so token ids don't depend on Python's hash seed;
  - `compute_perplexity` referenced an undefined variable, and now averages the loss over all predicted tokens;
  - evaluation uses non-overlapping windows, so each test token is predicted exactly once;
  - `utilities.py` saves figures to files instead of opening windows.

## Acknowledgements

I used Claude (Anthropic) while working on this assignment.
