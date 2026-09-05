# FS-SSA-GPT — Causal Spiking Self-Attention on Tiny Shakespeare and TinyStories

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22048497.svg)](https://doi.org/10.5281/zenodo.22048497)

This repository presents an **autoregressive** Transformer powered by **Softmax-Free Spiking Self-Attention** (FS-SSA). Query, key and value are quantised into at most K binary spikes by a few-spikes (FS) neuron, the softmax is removed entirely, and the architecture is evaluated across two distinct scales:

1. **Part 1 — Ablation & Mechanics (TinyShakespeare):** A ~2.7M parameter character-level model (6 layers) tested across 3 seeds to evaluate stability, causal normalization, and threshold dynamics against a matched full-precision control;

2. **Part 2 — Scaling Demonstration (TinyStories):** A ~25.1M parameter model (12 layers) using GPT-2 BPE subword tokenisation, evaluating whether the architecture scales smoothly to natural language generation;

Prior work in this line: [FS-SSA](https://github.com/LRMTV94/FS_Softmax_Free_Attention) established the same architecture as a classifier on synthetic point clouds and on LHC jets, where the plain neuron already matches the control.

**Key Findings:**

1. **Stability:** The spiking model is as reproducible across seeds as the full-precision control ($\sigma \approx 0.005$), showing zero training collapse or divergence across all runs.

2. **Convergence:** At 10k steps on TinyShakespeare, the FP32 control overfits while the spiking variants remain on a steady downward slope.

3. **Scalability:** At 25.1M parameters on TinyStories BPE, the Softmax-free `FS-SSA K=2 ± L` model achieves a validation loss of **$1.8238 \pm 0.0074$** (Perplexity **6.20**), operating directly in the same loss regime as dense FP32 transformers ($\sim 1.70 - 1.90$) reported in the literature.

---

## Why the autoregressive case is different

Three things change relative to the classifier, and all three are forced rather than chosen.

**BatchNorm cannot be used on Q/K/V.** `BatchNorm1d` over `(B, C, T)` pools statistics over time as well as batch, so in a causal model the statistics at position *t* would include future tokens: a direct leak. RMSNorm normalises over the channel dimension only. As a side effect it also (finally!) removes the running-statistics discrepancy that dominated the classifier results, since RMSNorm keeps no buffers at all.

**The row normalisation becomes causal.** Without a softmax the attention rows do not sum to 1 and must be divided by the number of attended keys. In a causal model position *t* attends to *t+1* keys, not to a constant, so the divisor is `arange(1, T+1)`. Dividing by a constant would crush the beginning of every sequence.

**The threshold scale is re-measured, not inherited.** `qk_scale = 0.25` was calibrated for a BatchNorm-ed input; the spread after RMSNorm is different. The script probes the pre-activation std at initialisation and derives both scales from it: here 1.000 for Q/K/V and 0.271 for the MLP pre-activation, giving `qk_scale = 0.750` and `mlp_scale = 0.271`. This matters because the resolved
input window is equal to:

```
window = [0, s·(2 − 2^-(K-1)))     →  2s  as K → ∞
```

with a hard ceiling at `2s`: raising K refines the quantisation step but can never widen the range, so a badly chosen `s` cannot be repaired with more spike levels. At `s = 0.750` and K=2, 13% of channels saturate.

### A prediction about the sign

In every classifier ablation the signed ON/OFF pair was measurable only at K=1 and cost roughly twice the spikes. Before running this experiment there was a mechanistic reason to expect otherwise here:

> With non-negative Q and K every causal logit is ≥ 0, so a token can be weighted less but never **suppressed**. The sign is what restores suppression.

**The prediction is not supported (with the current budget).** Adding the sign makes validation loss *worse* by +0.048 nats [+0.027, +0.069] against the plain neuron, at 1.6× the spikes, and the penalty is present at every point of the training curve rather than only at the end. Whatever suppression the sign restores, it does not pay for itself at this size. It is recorded here because it was stated in advance.

However, two critical observations nuance this initial finding:

1. **Ongoing Convergence:** Unlike the FP32 control (which overfitted and turned upward around step 8250), all spiking variants, including the signed ones, were still steadily descending at step 10 000, meaning this gap is evaluated before convergence.

2. **The Rescue by Learnable Ladders (`± L`):** When the signed neuron is paired with learnable channel thresholds (`K=2 ± L`), the network dynamically prunes noise: it recovers −0.019 nats, **lower attention spikes by 20%** (0.846 → 0.680 spikes/token), and produces the highest qualitative fidelity in generating distinct character dialogue tags (as detailed in the results below).

---

## Part 1: Results on TinyShakespeare (2.7M params)

Character-level, vocabulary 65, `block = 256`, `d_model = 192`, 6 heads, 6 layers, dropout 0.2, AdamW, 10 000 iterations, 3 seeds, batch 64.

<div align="center">

| Configuration | Best val loss | Perplexity | @ iter | Final val | Params | Attn spikes | MLP spikes | Minutes |
|---|---|---|---|---|---|---|---|---|
| **softmax + GELU** (control) | **1.4469 ± 0.0050** | **4.25** | 8917 | 1.4571 | 2,728,704 | — | — | 10 |
| FS-SSA K=2 | 1.5562 ± 0.0052 | 4.74 | 8917 | 1.5590 | 2,732,160 | 0.523 | 0.173 | 16 |
| FS-SSA K=2 ± | 1.6042 ± 0.0093 | 4.97 | 8917 | 1.6088 | 2,732,160 | 0.846 | 0.175 | 18 |
| FS-SSA K=2 ± L | 1.5851 ± 0.0049 | 4.88 | 8917 | 1.5924 | 2,794,368 | 0.680 | 0.168 | 21 |

</div>

### Convergence Slopes and Stability

Fitting a slope to the averaged validation curve over the last 2500 iterations:

<div align="center">

| Configuration | val @7500 | val @10000 | slope [nats / 1000 it] | Seed Spread ($\sigma$) |
|---|---|---|---|---|
| softmax + GELU | 1.4563 | 1.4571 | **+0.0017** (overfitting) | 0.0050 |
| FS-SSA K=2 | 1.5706 | 1.5590 | −0.0043 (still improving) | 0.0052 |
| FS-SSA K=2 ± L | 1.5973 | 1.5924 | −0.0020 (still improving) | 0.0049 |

</div>

The gap between control and spiking lost 77% of its initial value over 9000 iterations and was still shrinking when training stopped.


## Generated Text Samples

**These samples are illustrative, not a measurement.** They come from the weights at the final iteration, whereas the loss reported above is the minimum over training, so the model that produced this text is not the model those numbers describe. Sampling used temperature `t=0.8` and top-k 200, from an empty context. Read them to see whether the spiking model produces the same *kind* of output as the control, not to rank the two, at a 15% perplexity difference, ranking is what the table is for.

**softmax + GELU** (control, seed 2, val 1.4416)

```
But touch the field of the vows of the book,
which then yet about-shining words can never
Be a place would do the lid that die of her bodies
And so protect in the public of subjects.
To do the truth, and be so?
Go we to this soul's faults? For thine own side,
Of the court-lady stocks, being a straight of good
To keep our courties that would they be oath.

ROMEO:
I have the world, have loved thee them from the brother,
And these thy state all my purpose.

MERCUTIO:
Sirrah, sir, come.
```

**FS-SSA K=2** (best spiking configuration, seed 2, val 1.5505)

```
I cannot be no saint that you have broke you not?

STANLEY:
Why, then? What made you to hear you?

POMPEY:
Why, my lord.

POMPEY:
Good sir, and much you have well not be as those
father, you shall be so well and for ever syou.

POMPEY:
What's a gone?

LUCIO:
A know she name of her may but you will so out bencer.

BOPET:
Here, mean, what now have a marriage?
```

**FS-SSA K=2 ± L** (seed 2, val 1.5842)

```
I have should be and the favour'd mind her father.

KING EDWARD IV:
Now, bring the man, heart my soul love worls,
Where I have any the shame fairs thy word.

DUKE OF AUMERLE:
And for thy dispirate vass my blood.

KING EDWARD IV:
By I warrant thou from thy Edward's death,
My bright mayor arms and thy book being thee;
And should make was thy England's place three,
Art thou of thee and these blood mvirting and steel.
```


Both arms learn the same structure: speaker names in capitals followed by a colon, line breaks at roughly iambic length, dialogue alternating between speakers, and a vocabulary that is mostly real English with plausible malformations at the edges. Character names are drawn from the right plays and stay internally consistent within a passage.

Crucially, a distinct qualitative trade-off emerges between the spiking variants:

- **FS-SSA K=2** (plain) yields the lowest raw validation loss among spiking models, capturing simple dialogue exchanges (`STANLEY:`, `POMPEY:`).
  
- **FS-SSA K=2 ± L** (signed + learnable thresholds), despite a slightly higher loss (+0.03 nats over plain K=2), exhibits **the highest structural and dramatic fidelity**. It consistently generates complex, multi-speaker scenes featuring major historical and play-specific characters (`KING EDWARD IV:`, `DUKE OF AUMERLE:`, `FRIAR LAURENCE:`, `PAULINA:`).

While all spiking samples contain somewhat more invented words (`bencer`, `mvirting`, `syou`) and drift out of syntax sooner than the FP32 control (reflecting the 11–15% perplexity gap at this 10k budget), the combination of signed suppression and channel-level threshold adaptation in `FS-SSA K=2 ± L` preserves high-level character roles and dialogue hierarchy with remarkable fidelity, while **lowering attention spiking activity by 20%**.

The full set, one per configuration and seed, is in `results/fsssa_gpt_samples.txt`.

---

## Part 2: Scaling to TinyStories BPE (25.1M params)

To evaluate whether the architecture scales to natural language, we scaled the `FS-SSA K=2 ± L` configuration to **25.1M parameters** on the **TinyStories** dataset using **GPT-2 BPE subword tokenisation** (vocab 50,257). The configuration's properties used are: `block = 256`, `d_model = 384`, 6 heads, 12 layers, batch size 64 (accumulated), 5000 iterations, 2 seeds.

<div align="center">

| Configuration | Best val loss | Perplexity | @ iter | Params | Attn spikes | MLP spikes | VRAM |
|---|---|---|---|---|---|---|---|
| **FS-SSA K=2 ± L** (Seed 0) | 1.8185 | 6.16 | 4750 | 25,123,584 | 0.680 | 0.065 | 9.3 GB |
| **FS-SSA K=2 ± L** (Seed 1) | 1.8290 | 6.23 | 5000 | 25,123,584 | 0.681 | 0.065 | 9.8 GB |
| **FS-SSA K=2 ± L** (Seed 2) | 1.8015 | 6.09 | 4500 | 25,123,584 | 0.681 | 0.065 | 9.8 GB |
| **MEAN ± STD** | **1.8163 ± 0.0139** | **6.16** | **4750** | **25,123,584** | **0.681** | **0.065** | — |

</div>


## Generated Text Samples

**These samples are illustrative, not a measurement.** They come from the weights at the final iteration, whereas the loss reported above is the minimum over training, so the model that produced this text is not the model those numbers describe. Read them as a sort of guideline (there’s no direct comparison with a vabilla trsformer, due to budget constraints):

**FS-SSA K=2 +/- L** (seed 0, val 1.8185 , ppl 6.16)

```
<|endoftext|>. He felt sad and lonely.
Tim and Mia looked at the billboard. They did not know what to do. They wanted to buy another picture. They tried to make a new picture. They saw the picture of a rainbow. It was a picture of the sun. They smiled.
<|endoftext|>

Tim and Sam were best friends. They liked to play with their toys and make funny noises. One day, they found a big box under a table. They opened the box and saw many toys and games.
"Wow, look at these toys!" Tim said. "Can we play with them?"
"Yes, you can," Sam said. "But I found them first. They are big and red and has a flag. I can put them in the box for Mia."
"No, we can't," Tim said. "We can play with them and trucks. They are my toys. We can share them and share them with each other. Mommy says we can play with them later."
"But we don't have any friends. They are a huge toy. We can have more fun with them." Sam said. He did not listen to Sam.
Tim thought Sam was stupid. He did not care that Sam was mean. He said, "No, Sam. You are bad. You don't want to play with them. They are fun." He pulled the toys and threw them on the floor."
```

**FS-SSA K=2 +/- L** (seed 1 , best val 1.8290 , ppl 6.23)

```
<|endoftext|> got to give up. She handed him a special knife. It was a gift from the top. She held it up really high in her hand.
Mimi said, "Wow! This is my reward!"
The two of them hugged each other and decided to keep being so thoughtful. Mimi and Pimi had a lot of fun on the day together. They were never bored.
<|endoftext|>

Once upon a time, there was a little girl named Lucy. She was three years old. One day, her mommy asked her to help her carry a sack. Lucy was very excited and asked, “Mom, what is this?”
Her mom said “It’s an address.”
Lucy put on her blue shirt and asked, “Mommy, can I help.”
Her mom smiled and said, “That’s a great idea, Lucy.”
Lucy’s eyes lit up. She had never heard of an open door before. She smiled and hugged her mom.
“What do you need us to do?” she asked. Her mom smiled and said, “I’m going to put something, but it’s not safe.”
Lucy was excited. “What kind of surprise?” she asked.
Her mom smiled and said, “It’s an
```
**FS-SSA K=2 +/- L** ( seed 2 , best val 1.8065 , ppl 6.09)

```
<|endoftext|> decided he had just left the box in the park.
When John went to the park he found the box and he decided to play with it. He was so excited! He ran over to the box and grabbed it. He opened it and saw lots of colorful pictures inside. He smiled. He ran around and played with his new toys.
John was having so much fun. He pretended that the box was a very interesting place to hide and seek inside. He even had so much fun! He was so happy that he forgot how much he was playing with the box and his fun day.
<|endoftext|>

The farmer and the cow were going to the beach. The cow was looking for something special. The farmer went to the shed and got an idea. He wanted to make a new sandcastle. He grabbed a bucket and started to work. 
The farmer was very careful and was able to clean the sand. He worked carefully to make the sandcastle. He was so proud of himself and started to build the beach. After he finished, he added lots of sand and made a sandcastle. 
The farmer was very proud of his work. He was proud that he could make the best sandcastle ever! He had built a beautiful time and he was very proud of his work.
<|endoftext|>

Once upon a time, there was a little boy who was very polite. He liked to observe the things
```

At 25.1M parameters with BPE tokenisation, spelling malformations disappear entirely. The FS-SSA model generates coherent narrative arcs, multi-speaker dialogues with quotes and attributions, character consistency, and emotional actions.

The full set, one per configuration and seed, is in `results/fsssa_tinystories_samples.txt`.


**Literature Comparison:** Published dense FP32 transformers of equivalent size (~28M) on TinyStories achieve validation loss around **~1.70 – 1.90** (Eldan & Li, 2023). The Softmax-free spiking model $K=2 \pm L$ operates directly in the same performance regime while using a softmax-free attention.

---

## Usage

One script, no packages, no subdirectories. Both run on Colab with a GPU (T4 or A100).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib numpy

python fs_ssa_gpt.py     # trains, writes fsssa_gpt_state.json (Tiny Shakespeare) 
python fsssa_gpt_bpe.py     # trains, writes fsssa_gpt_state.json (TinyStories)
```

To repeat the run at a longer budget, change `MAX_ITERS`; the cosine schedule adapts to it.

---

## Future work

Given that the architecture demonstrated solid numerical stability across seeds ($\sigma \approx 0.005$, with zero diverged or collapsed runs), the following directions represent the immediate next steps:

1. **Extended iteration budgets:**
   Training for 50 000–100 000 steps to measure the true asymptotic convergence limit of the spiking attention, testing whether the steady downward slope observed at 10k steps ultimately plateaus near or at the control loss.
   
2. **Scaling to 50–200M parameters:**
   Moving from the 25–30M architectures to 100-200M parameters architectures, and evaluating actual 25M acrchietecture on the benckmark **WikiText-2**.
   
3. **Causal memory decay ($\gamma$):**
   Incorporating a learnable or fixed decay factor ($\gamma $) into the causal state accumulation ($S_t = \gamma S_{t-1} + K_t^T V_t$) to introduce recency bias and prevent state saturation over very long sequence contexts ($T \ge 4096$).

---

## References

- Stöckl & Maass, *Optimized spiking neurons can classify images with high
  accuracy through temporal coding with two spikes*, Nature Machine Intelligence
  3, 230–238 (2021).

- Zhou et al., *Spikformer: When Spiking Neural Network Meets Transformer*,
  ICLR 2023.

- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) — the baseline
  architecture and hyperparameters this follows, and the source of the dataset.

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.

## Citation and Acknowledgements

If you use this codebase or the FS-SSA architecture in your research, please cite:

```bibtex
@software{FS_Softmax_Free_Attention_2026,
  author    = {Lo Russo Matteo Vito},
  title     = {Few-Spikes Transformer with Spiking Self-Attention},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22048497},
  url       = {https://doi.org/10.5281/zenodo.22048497}
}
```

Thanks for your.... Attention! 😄

---
