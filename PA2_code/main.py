import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import (TransformerEncoder, TransformerDecoder, FeedForwardClassifier, EncoderClassifier,
                         count_parameters)
from utilities import Utilities

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
ff_hidden = 100  # Hidden size of the feedforward layer inside each transformer block (encoder and decoder)


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Maximum number of batches used to evaluate perplexity (the eval sets used here are smaller, so they are evaluated in full)


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

DATA_DIR = "speechesdataset"
PLOT_DIR = "plots"
SPEAKERS = {"obama": "Barack Obama", "wbush": "George W. Bush", "hbush": "George H. W. Bush"}

# Sentences used for the attention-map sanity checks (Parts 1 and 2)
SANITY_SENTENCES = [
    "It depends on us, on the choices we make, particularly at certain inflection points in history.",
    "We will not tire, we will not falter, and we will not fail.",
]


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = sorted(os.listdir(directory))
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)  # the model returns (logits, attention maps)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=eval_iters):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Perplexity = exp(mean cross-entropy per predicted token), averaged over all tokens in up to eval_iters batches.
    """
    decoderLMmodel.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            loss, _ = decoderLMmodel(X, Y)  # the model computes the mean cross-entropy loss
            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
    decoderLMmodel.train()
    return math.exp(total_loss / total_tokens)


def attention_statistics(model, data_loader, causal, max_batches=50):
    """Summarizes every head over real data, averaged over query positions with at least 2 visible keys:
    normalized entropy (1 = uniform over the visible keys, 0 = all weight on one key), the largest weight in the
    row, and the weight on the query's own position, on the previous position and on the first position."""
    model.eval()
    stats = None
    with torch.no_grad():
        for b, (X, _) in enumerate(data_loader):
            if b >= max_batches:
                break
            X = X.to(device)
            _, maps = model(X)
            T = X.shape[1]
            pos = torch.arange(T, device=device)
            if causal:
                n_keys = (pos + 1).expand(X.shape[0], T).float()                    # row i sees keys 0..i
                valid = n_keys >= 2
            else:
                length = (X != 0).sum(dim=1, keepdim=True)                           # real tokens per sequence
                n_keys = length.expand(-1, T).float()
                valid = (pos[None, :] < length) & (n_keys >= 2)
            if stats is None:
                stats = torch.zeros(len(maps), 5)
                counts = torch.zeros(5)
            prev_idx = (pos - 1).clamp(min=0)
            masks = [valid, valid, valid, valid & (pos[None, :] >= 1), valid]   # "previous" needs a position >= 1
            for h, A in enumerate(maps):
                ent = -(A * torch.log(A.clamp(min=1e-12))).sum(-1) / torch.log(n_keys.clamp(min=2))
                vals = [ent, A.max(-1).values, A.diagonal(dim1=-2, dim2=-1),
                        A[:, pos, prev_idx], A[:, :, 0]]
                for s, (v, m) in enumerate(zip(vals, masks)):
                    stats[h, s] += v[m].sum().item()
            counts += torch.tensor([m.sum().item() for m in masks], dtype=torch.float)
            uniform_sum = (uniform_sum if b else 0.0) + (1.0 / n_keys)[valid].sum().item()
    model.train()
    uniform = uniform_sum / counts[0].item()
    stats /= counts
    print(f"{'layer':>5} {'head':>4}  {'norm. entropy':>13}  {'max weight':>10}  {'self':>6}  {'previous':>8}  {'first':>6}")
    for h in range(stats.shape[0]):
        layer, head = divmod(h, n_head)
        print(f"{layer + 1:>5} {head + 1:>4}  " + "  ".join(f"{v:>{w}.3f}" for v, w in zip(stats[h].tolist(), (13, 10, 6, 8, 6))))
    print(f"(uniform attention over the visible keys would give entropy 1.000 and about {uniform:.3f} "
          f"for each of the weight columns)")
    return stats


def attention_sanity_checks(tokenizer, model, tag):
    """Row-sum check and attention-map plots for each sanity sentence."""
    util = Utilities(tokenizer, model)
    for s, sentence in enumerate(SANITY_SENTENCES, start=1):
        print(f"\nSanity check, sentence {s}: {sentence!r}")
        util.sanity_check(sentence, block_size, prefix=f"{tag}_s{s}_attention_map",
                          outdir=os.path.join(PLOT_DIR, f"{tag}_attention_maps"))
        path = os.path.join(PLOT_DIR, f"{tag}_attention_s{s}.png")
        util.plot_attention_grid(sentence, block_size, n_layer, n_head, path,
                                 title=f"{tag}: attention maps (row = query, column = key)")
        print(f"Saved all heads of all layers to {path}")


def run_part1(tokenizer, args):
    print("\n" + "=" * 70 + "\nPart 1: Transformer encoder + feedforward classifier, trained jointly\n" + "=" * 70)
    torch.manual_seed(args.seed)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(DATA_DIR, "train_CLS.tsv"))
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(DATA_DIR, "test_CLS.tsv"))
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    print(f"{len(train_CLS_dataset)} training and {len(test_CLS_dataset)} test segments")

    encoder = TransformerEncoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, ff_hidden,
                                 mask_padding=not args.no_pad_mask, init=args.init)
    classifier = FeedForwardClassifier(n_input, n_hidden, n_output)
    model = EncoderClassifier(encoder, classifier).to(device)
    print(f"Padding is {'ignored (masked in attention and in the mean)' if encoder.mask_padding else 'NOT masked'}")
    print(f"Number of parameters: encoder {count_parameters(encoder):,}, classifier {count_parameters(classifier):,}, "
          f"total {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    t0 = time.time()
    for epoch in range(epochs_CLS):
        total_loss, n = 0.0, 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        train_acc = compute_classifier_accuracy(model, train_CLS_loader)
        test_acc = compute_classifier_accuracy(model, test_CLS_loader)
        history.append((epoch + 1, total_loss / n, train_acc, test_acc))
        print(f"Epoch {epoch + 1:2d}: train loss {total_loss / n:.4f}  train acc {train_acc:6.2f}%  "
              f"test acc {test_acc:6.2f}%")
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Final test accuracy: {history[-1][3]:.2f}%")

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h[0] for h in history], [h[2] for h in history], "o-", label="train")
    ax.plot([h[0] for h in history], [h[3] for h in history], "s-", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Part 1: encoder + classifier accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "part1_accuracy.png"), dpi=150)
    plt.close(fig)

    print("\nAttention statistics of the trained encoder on test_CLS.tsv:")
    attention_statistics(model, test_CLS_loader, causal=False)
    attention_sanity_checks(tokenizer, model, "part1_encoder")
    return history, model


def loss_on_unseen_tokens(model, train_dataset, eval_loaders, tokenizer):
    """Splits each test set's loss between tokens that never occur in train_LM.txt (they are in the vocabulary
    because it is also built from train_CLS.tsv, or they are <unk>) and all other tokens."""
    seen = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    seen[train_dataset.data.to(device)] = True
    print("\nTest loss on tokens never seen in train_LM.txt vs the other tokens (nats/token):")
    print(f"{'test set':>18}  {'unseen tokens':>13}  {'loss unseen':>11}  {'loss others':>11}  {'share of total loss':>19}")
    model.eval()
    with torch.no_grad():
        for key, speaker in SPEAKERS.items():
            loss_sum = {True: 0.0, False: 0.0}
            count = {True: 0, False: 0}
            for X, Y in eval_loaders[key]:
                X, Y = X.to(device), Y.to(device)
                logits, _ = model(X)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                         reduction="none")
                unseen = ~seen[Y.reshape(-1)]
                for flag in (True, False):
                    loss_sum[flag] += loss[unseen == flag].sum().item()
                    count[flag] += (unseen == flag).sum().item()
            total = loss_sum[True] + loss_sum[False]
            print(f"{speaker:>18}  {100 * count[True] / (count[True] + count[False]):>12.2f}%  "
                  f"{loss_sum[True] / count[True]:>11.2f}  {loss_sum[False] / count[False]:>11.2f}  "
                  f"{100 * loss_sum[True] / total:>18.1f}%")
    model.train()


def run_part2(tokenizer, args):
    print("\n" + "=" * 70 + "\nPart 2: Decoder language model pretraining\n" + "=" * 70)
    torch.manual_seed(args.seed)

    inputfile = os.path.join(DATA_DIR, "train_LM.txt")
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Evaluation sets: non-overlapping windows, so every token is predicted exactly once
    eval_texts = {"train": lmtrainText}
    for key in SPEAKERS:
        with open(os.path.join(DATA_DIR, f"test_LM_{key}.txt"), 'r', encoding='utf-8') as f:
            eval_texts[key] = f.read()
    eval_loaders = {}
    for key, text in eval_texts.items():
        ds = LanguageModelingDataset(tokenizer, text, block_size, stride=block_size)
        eval_loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False)
        n_tok = len(ds.data)
        n_unk = (ds.data == tokenizer.stoi['<unk>']).sum().item()
        print(f"{key:>6}: {n_tok:6d} tokens, {len(ds)} windows, {100 * n_unk / n_tok:.2f}% <unk>")

    model = TransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, ff_hidden,
                               init=args.init).to(device)
    print(f"Number of parameters in the decoder: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    names = list(eval_loaders)
    header = f"{'iter':>5}  {'batch loss':>10}  " + "  ".join(f"{('ppl ' + k):>10}" for k in names)
    history = [(0, float("nan"), {k: compute_perplexity(model, eval_loaders[k]) for k in names})]
    print(header)
    print(f"{0:>5}  {'':>10}  " + "  ".join(f"{history[0][2][k]:>10.2f}" for k in names))
    t0 = time.time()
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss, _ = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % eval_interval == 0 or i + 1 == max_iters:
            ppl = {k: compute_perplexity(model, eval_loaders[k]) for k in names}
            history.append((i + 1, loss.item(), ppl))
            print(f"{i + 1:>5}  {loss.item():>10.4f}  " + "  ".join(f"{ppl[k]:>10.2f}" for k in names))
    print(f"Training time: {time.time() - t0:.1f}s")
    final = history[-1][2]
    print("Final perplexity: " + ", ".join(f"{k} {final[k]:.2f}" for k in names))

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k in names:
        label = "train" if k == "train" else f"test: {SPEAKERS[k]}"
        ax.plot([h[0] for h in history], [h[2][k] for h in history], "o-", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("perplexity (log scale)")
    ax.set_title("Part 2: decoder perplexity during pretraining")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "part2_perplexity.png"), dpi=150)
    plt.close(fig)

    loss_on_unseen_tokens(model, train_LM_dataset, eval_loaders, tokenizer)
    print("\nAttention statistics of the trained decoder on test_LM_obama.txt:")
    attention_statistics(model, eval_loaders["obama"], causal=True)
    attention_sanity_checks(tokenizer, model, "part2_decoder")
    return history, model


def main():
    global PLOT_DIR
    parser = argparse.ArgumentParser(description="CSE 256 PA2, Parts 1 and 2")
    parser.add_argument("part", nargs="?", default="all", choices=["part1", "part2", "all"],
                        help="part1: encoder + classifier; part2: decoder LM; all (default): both")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--no_pad_mask", action="store_true",
                        help="Part 1: let the encoder attend to padding and include it in the mean (for comparison)")
    parser.add_argument("--init", default="gpt2", choices=["gpt2", "default"],
                        help="gpt2: N(0, 0.02) weights (default); default: PyTorch's built-in initialization")
    parser.add_argument("--plot_dir", default=PLOT_DIR, help="where plots are saved (default: plots)")
    args = parser.parse_args()
    PLOT_DIR = args.plot_dir
    print(f"device: {device}, seed: {args.seed}, init: {args.init}")

    print("Loading data and creating tokenizer ...")
    texts = load_texts(DATA_DIR)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.part in ("part1", "all"):
        run_part1(tokenizer, args)
    if args.part in ("part2", "all"):
        run_part2(tokenizer, args)


if __name__ == "__main__":
    main()
