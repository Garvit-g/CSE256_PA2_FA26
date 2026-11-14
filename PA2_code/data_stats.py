"""Data statistics behind the Part 2 discussion of why the three politicians get different perplexities.

    python data_stats.py

For each test file it reports how much of it resembles the language-modeling training text (train_LM.txt):
  - which speakers train_LM.txt contains, found by matching its lines against the labeled classification files
  - the share of test words that never occur in train_LM.txt (they are in the vocabulary, because the vocabulary
    also includes train_CLS.tsv, but the decoder never saw them as training targets)
  - the share of test bigrams that also occur in train_LM.txt
  - the perplexity of an add-one smoothed unigram model trained on train_LM.txt
"""
import math
import os
from collections import Counter

from nltk.tokenize import word_tokenize

from main import DATA_DIR, SPEAKERS, load_texts
from tokenizer import SimpleTokenizer


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    tokenizer = SimpleTokenizer(" ".join(load_texts(DATA_DIR)))
    label_of = {}
    for name in ("train_CLS.tsv", "test_CLS.tsv"):
        for l in lines(os.path.join(DATA_DIR, name)):
            label, text = l.split("\t")
            label_of[text.strip()] = label
    names = {"0": "Barack Obama", "1": "George W. Bush", "2": "George H. W. Bush"}

    train_lines = lines(os.path.join(DATA_DIR, "train_LM.txt"))
    who = Counter(names.get(label_of.get(l), "not in the labeled files") for l in train_lines)
    print(f"train_LM.txt: {len(train_lines)} lines; speaker of each line, matched against the labeled files:")
    for k in list(names.values()) + ["not in the labeled files"]:
        print(f"  {k:>25}: {who.get(k, 0)}")

    with open(os.path.join(DATA_DIR, "train_LM.txt"), encoding="utf-8") as f:
        train_text = f.read()
    train_words = set(word_tokenize(train_text))
    train_ids = tokenizer.encode(train_text)
    unigram, N, V = Counter(train_ids), len(train_ids), tokenizer.vocab_size
    train_bigrams = set(zip(train_ids, train_ids[1:]))
    train_set = set(train_lines)

    print(f"\n{'test set':>18}  {'tokens':>6}  {'<unk>':>6}  {'unseen in train_LM':>18}  {'bigrams seen':>12}  "
          f"{'unigram ppl':>11}  {'lines in train_LM':>17}")
    for key, speaker in SPEAKERS.items():
        path = os.path.join(DATA_DIR, f"test_LM_{key}.txt")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)
        words = word_tokenize(text)
        unk = sum(i == tokenizer.stoi["<unk>"] for i in ids) / len(ids)
        unseen = sum(w not in train_words for w in words) / len(words)
        seen_bigrams = sum(b in train_bigrams for b in zip(ids, ids[1:])) / (len(ids) - 1)
        unigram_ppl = math.exp(-sum(math.log((unigram[i] + 1) / (N + V)) for i in ids) / len(ids))
        overlap = sum(l in train_set for l in lines(path))
        print(f"{speaker:>18}  {len(ids):>6}  {100 * unk:>5.2f}%  {100 * unseen:>17.2f}%  {100 * seen_bigrams:>11.1f}%  "
              f"{unigram_ppl:>11.1f}  {overlap:>13}/250")


if __name__ == "__main__":
    main()
