
import os

import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _input(self, sentence, block_size):
        wordids = self.tokenizer.encode(sentence)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        device = next(self.model.parameters()).device
        return torch.tensor(padded_sentence, dtype=torch.long, device=device).unsqueeze(0), min(len(wordids), block_size)

    def sanity_check(self, sentence, block_size, prefix="attention_map", outdir="plots"):
        """Checks that every attention row sums to 1 and saves one heatmap per head per layer."""
        os.makedirs(outdir, exist_ok=True)
        # Encode the sentence and pad it to block_size
        input_tensor, _ = self._input(sentence, block_size)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the model; only the attention maps are needed
        self.model.eval()
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor)
        self.model.train()

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        all_ok = True
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1).cpu()
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                all_ok = False
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)
            plt.title(f"Attention Map {j + 1}")

            # Save the plot
            plt.savefig(os.path.join(outdir, f"{prefix}_{j + 1}.png"))
            plt.close(fig)
        if all_ok:
            print(f"Normalization test passed: every row of all {len(attn_maps)} attention maps sums to 1")
        return all_ok

    def plot_attention_grid(self, sentence, block_size, n_layer, n_head, path, title=""):
        """All heads of all layers in one figure (rows = heads, columns = layers), labelled with the real tokens.
        Padding positions are cropped out. Row = query position, column = key position."""
        input_tensor, n = self._input(sentence, block_size)
        self.model.eval()
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor)
        self.model.train()
        tokens = [self.tokenizer.itos.get(i, "<unk>") for i in input_tensor[0, :n].tolist()]

        fig, axes = plt.subplots(n_head, n_layer, figsize=(3.6 * n_layer, 3.6 * n_head), squeeze=False,
                                 gridspec_kw={"wspace": 0.45, "hspace": 0.4})
        for j, attn_map in enumerate(attn_maps):
            layer, head = divmod(j, n_head)
            ax = axes[head][layer]                                        # rows = heads, columns = layers
            im = ax.imshow(attn_map[0, :n, :n].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(n))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(tokens, fontsize=6)
            ax.set_title(f"layer {layer + 1}, head {head + 1}", fontsize=9)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="attention weight")
        if title:
            fig.suptitle(title)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")   # vector version for the report
        plt.close(fig)
        return tokens, [m[0, :n, :n].cpu() for m in attn_maps]
