# Few-Spikes Transformer with Spiking Self-Attention (FS-SSA)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22048497.svg)](https://doi.org/10.5281/zenodo.22048497)

Can a transformer whose **activations *and* attention are both spiking** compete with a full-precision one?

Few-spikes (FS) neurons quantise an activation into at most K binary spikes. They carry no membrane state and no notion of time (see below)  in contrast to LIF neurons. Two earlier repositories established that FS activations cost nothing inside otherwise standard networks: [Few_Spikes_SNN](https://github.com/LRMTV94/Few_Spikes_SNN) for MLPs and CNNs, and [Few_Spikes_Transformer](https://github.com/LRMTV94/Few_Spikes_Transformer) for a transformer where the attention softmax was explicitly left at full precision. **This repository removes that last exception.**

**Answer: yes, and the plain neuron is enough.** On a synthetic ring-counting task the simplest configuration (binary spikes, no sign, fixed thresholds) is statistically indistinguishable from the matched full-precision control. On real
LHC jets the deficit is small but measurable, and it shrinks as the number of spike levels grows.

<div align="center">

| | synthetic rings (19 seeds) | JetNet g/q (10 seeds) |
|---|---|---|
| softmax_bn + GELU (control) | **79.39% ± 1.28** | **79.02% ± 0.39**, AUC 0.8719 |
| FS-SSA K=1, plain | 79.34% ± 1.11 | 78.26% ± 0.38, AUC 0.8653 |
| FS-SSA K=3, plain | 78.41% ± 1.00 | 78.72% ± 0.39, AUC 0.8692 |
| paired deficit, best plain | **+0.06 pp** [−0.52, +0.63] | **+0.30 pp** [+0.04, +0.56] |

</div>

Neither the signed ON/OFF pair nor the learnable threshold ladder is required to reach this. Both are measured below; on the synthetic task neither has a detectable effect, and on JetNet only the sign at K=1 does.

---

## A correction to the previous revision

An earlier version of this repository reported a different result: that a plain FS-SSA trains erratically, with a seed-to-seed spread of up to ±7 pp, and that the signed and learnable variants were what removed the instability. That
conclusion does not survive a change in how the trained models were evaluated, and this section records the correction in full.

**The discrepancy was in evaluation, not in training.** `BatchNorm` estimates its inference-time statistics with an exponential moving average whose default momentum retains roughly ten batches of history. While the weights are still moving those buffers describe a network that no longer exists.

Evaluating in `eval()` therefore normalises the current activations with slightly stale parameters. For a smooth activation (GeLU) the resulting displacement is proportional and largely absorbed while for an hard threshold (FS) it's not: a channel sitting just above its threshold falls silent entirely, and because Q, K and V are all quantised and then multiplied together, the perturbation compounds.

The remedy **does not touch** the model. After training, the buffers are reset and recomputed exactly over the training set with a cumulative rather than exponential average; the weights are untouched and only the reading changes. Both
readings are now reported side by side (`accuracy` and `accuracy_ema` in the result files). The effect is concentrated in the variance:

<div align="center">

| seed σ, SSA configurations only | EMA reading | exact statistics |
|---|---|---|
| synthetic rings | 2.79 – 9.19 | **0.73 – 1.20** (control 1.28) |
| JetNet g/q | 0.79 – 2.26 | **0.25 – 0.46** (control 0.39) |

</div>

The discrepancy is present for every configuration including the full-precision control, and its size depends on both configuration and dataset: on the synthetic task the control shifts by +0.21 pp against +2.3 to +7.0 pp for the spiking
variants, whereas on JetNet the control is the most affected of all (+1.91 pp). What is consistent across both datasets is the collapse of the seed spread, by a factor between three and eight, once the statistics are exact.

**The variants did reduce the artefact, which is why they appeared to reduce instability.** This is worth stating plainly, because it is the reason the earlier conclusion was reached in good faith. 

A signed encoder is antisymmetric and locally smoother than a one-sided threshold and per-channel learnable thresholds can drift toward the actual pre-activation distribution instead of remaining pinned at $2^{-k}$. Both therefore damp the sensitivity to stale statistics, and their effect is measurable: averaged over configurations, the sign reduces the EMA-to-exact gap from +4.70 to +3.55 pp on the synthetic task and from +1.20 to +0.69 pp on JetNet, and on JetNet it also halves the seed spread under the old reading (1.52 → 0.98). So, the earlier tables are thus a faithful record of the suppression of an evaluation artefact rather than a property of the architecture, and once the artefact is removed for every 

The same reading explains a second observation reported earlier, that adding a cosine learning-rate schedule removed the bimodality. A schedule that anneals the learning rate towards zero leaves the weights nearly stationary over the
final epochs, which is precisely the condition under which an exponential moving average converges to the correct statistics. The schedule did not stabilise training so much as make the existing evaluation valid.

Three further corrections were made to the code at the same time, none of which changes the reported conclusions but all of which affect reproducibility:

- **The two FS scales were conflated.** A single `scale` was applied to the thresholds and reset in the fixed neuron but to the thresholds, reset *and* readout weights in the learnable one, so the two were not the same function at
  initialisation. They are now separated into `threshold_scale`, which sets the resolved input window, and `readout_scale`, which sets the output alphabet and has no effect on the spike pattern. An assertion in `src/neurons.py` now
  checks that the learnable variant reproduces the fixed one exactly at initialisation.

- **The MLP neuron did not receive its configuration.** `make_activation` was called with three arguments against a seven-argument signature, so the feed-forward neuron silently fell back to unit thresholds and a fixed ladder while the
  attention neurons were configured. The `L` ablation consequently applied to the attention only. The channel count must also be the MLP hidden width rather than `d_model`.

- **`signed` was dropped in the JetNet script.** In `Jetnet_Experiment.py` the block constructor called `make_attention` with eight arguments against a nine-argument signature, omitting the sign flag. The `±` rows of that script did not
  test what their label stated. A regression check now counts the ON/OFF pairs actually instantiated in the model.

Spike counts were also being read from the final evaluation batch only and included padded positions; they are now accumulated across the whole evaluation pass and masked. On JetNet, where padding is about 4% of tokens, this changes the
reported rates by less than 0.01 spikes per neuron.

All numbers in this README come from the result files in `results/`, produced with the corrected code. Figures and tables from the previous revision should not be compared against them.

---

## Background: the few-spikes neuron and its two variants

A standard FS neuron replaces a continuous activation with K discrete steps. At each step it either spikes or stays silent, and the output is the weighted sum of those spikes:

```
out = Σ_k s_k · d_k        s_k ∈ {0, 1},  T_k = h_k = s·2^-k,  d_k = r·2^-k
```

This is a binary expansion of the activation value: with K levels it represents 2^K distinct outputs, all of them **non-negative**. That is fine in a feed-forward branch, where FS replaces a ReLU, but it is a real limitation inside the attention, where Q·K is a *signed* similarity.

The two scales play different roles and must not be tied together. The thresholds and reset decide *which* spikes fire and *when*, so scaling them by `s` sets the input window the neuron resolves:

```
window = [0, s·(2 − 2^-(K-1)))        →  2s  as K → ∞
LSB    = s·2^-(K-1)
```

The ceiling at `2s` is worth noting: raising K refines the step but can never widen the range. With `s = 0.25` and a BatchNorm-ed input, roughly 31% of channels clip regardless of K. The readout weights `d` have no influence at all on the spike pattern, so scaling them by `r` sets the output alphabet, the neuron gain is `r/s`, and the energy depends on `s` alone.

### Signed FS: ON/OFF populations (`±`)

Two neurons of opposite polarity, in the manner of ON/OFF cells in the retina and of the polarity channels of event-based sensors:

```
              ┌── FS_on(x)  ──┐
    x  ───────┤               ├──►  out = FS_on(x) − FS_off(−x)
              └── FS_off(−x) ─┘
```

<div align="center">

| input | −2.0 | −1.0 | 0.0 | +1.0 | +2.0 |
|---|---|---|---|---|---|
| plain FS | 0.00 | 0.00 | 0.00 | +1.00 | +1.88 |
| signed FS | −1.88 | −1.00 | 0.00 | +1.00 | +1.88 |

</div>

Correlation with the input rises from 0.85 to 0.99. The cost is two populations, so at equal spike budget **signed FS with K levels sits beside plain FS with 2K levels**: the comparison is sign *versus* resolution, not sign *on top of*
resolution.

### Learnable thresholds, per channel (`L`)

A naive attempt to let the network adapt the neuron would be adding a learnable gain factor in front of the activation. That turns out to be **mathematically redundant**: BatchNorm already contains a per-channel affine scale ($\gamma$),
and multiplying its output by $g$ is mathematically identical to dividing all thresholds by $g$. Empirical tests confirmed this: an explicit gain factor changed nothing.

What BatchNorm *cannot* do is alter the **relative shape** of the threshold ladder, which is fixed at $2^{-k}$. The variant that adds true degrees of freedom is to learn the threshold ladders themselves on a per-channel basis:

```
T, d, h = reverse_cumsum(softplus(raw))    positive and strictly decreasing by
                                           construction, initialised to
                                           reproduce s·2^-k exactly
```

Two design choices make this parameterisation stable:

* **Monotonicity and positivity by construction.** Unconstrained parameters degenerate within a few epochs as thresholds cross or turn negative. The reverse cumulative-softplus form guarantees $T_0 > T_1 > \dots > T_{K-1} > 0$ regardless
  of the optimiser state.

* **Exact baseline initialisation.** Initialising through the inverse softplus makes the neuron reproduce the fixed geometric ladder exactly at epoch 0, so any difference in outcome is attributable to gradient adaptation rather than to a
  different starting point. This is asserted in `src/neurons.py` and was the property violated by the scale bug described above.

## Spiking Self-Attention (SSA)

Following [Spikformer](https://arxiv.org/abs/2209.15425), the softmax is **removed entirely** rather than approximated. Spike-form Q and K are non-negative, so their dot product needs no normalisation in sign, and the
attention reduces to accumulations over spike patterns.

Two details are easy to get wrong, and both were found by measurement rather than by reasoning:

- **Row normalisation.** Without a softmax the attention rows do not sum to 1, so `att @ v` is a *sum* over the sequence rather than a weighted mean. With ~120 tokens the attention output came out at **124× the residual stream**, swamping
the residual branch. Dividing by the number of valid keys restores it to 1.0×.

- **Threshold scale.** The default FS thresholds (1, ½, ¼, …) sit above the typical magnitude of the projections. Left unscaled, `QK^T` collapses to **2.8% non-zero entries, with 52% of tokens attending to nothing**. With `qk_scale = 0.25`
the matrix stays at ~45% non-zero and no rows die.

Q and K carry the sign; **V is left non-negative**, since it is a value being transported, not a similarity.

## Results

Both sweeps use `d_model = 64`, `depth = 2`, 4 heads, `width = 1.2`, `qk_scale = 0.25`, `mlp_scale = 1.0`, `readout_scale = 1.0`, Adam at `lr = 1e-3`, and no learning-rate schedule. The two tasks differ in almost everything else and are
not comparable to each other; each is read against its own paired control.

### Synthetic temporal ring counting

![Sample events](figures/sample_events.png)

Counting Cherenkov-like rings in a time-ordered sequence of hits `(x, y, t)` on a sparse sensor grid, in the hardest regime (`time_sep = 0`: no temporal information at all, so the model works from geometry alone). 19 seeds, 20 epochs,
5000 train / 2000 test events.

<img src=figures/ssa_comparison_colab.png width="3000">

<div align="center">

| Configuration | Accuracy | Params | Attn spikes | MLP spikes |
|---|---|---|---|---|
| softmax_bn + GELU | **79.39% ± 1.28** | 109,123 | — | — |
| FS-SSA K=1 | **79.34% ± 1.11** | 109,123 | 0.38 | 0.05 |
| FS-SSA K=1 ± | 79.01% ± 1.09 | 109,123 | 0.64 | 0.05 |
| FS-SSA K=1 L | **79.50% ± 0.73** | 111,811 | 0.38 | 0.05 |
| FS-SSA K=1 ± L | 78.99% ± 1.11 | 112,579 | 0.64 | 0.05 |
| FS-SSA K=2 | 78.58% ± 0.88 | 109,123 | 0.77 | 0.06 |
| FS-SSA K=2 ± | 78.56% ± 1.17 | 109,123 | 1.31 | 0.17 |
| FS-SSA K=2 L | 78.36% ± 1.11 | 114,499 | 0.77 | 0.06 |
| FS-SSA K=2 ± L | 78.65% ± 1.07 | 116,035 | 1.31 | 0.17 |
| FS-SSA K=3 | 78.41% ± 1.00 | 109,123 | 1.15 | 0.11 |
| FS-SSA K=3 ± | 78.66% ± 1.07 | 109,123 | 1.98 | 0.32 |
| FS-SSA K=3 L | 78.45% ± 1.20 | 117,187 | 1.16 | 0.11 |
| FS-SSA K=3 ± L | 78.87% ± 1.05 | 119,491 | 1.98 | 0.31 |

</div>

**The binary configuration already matches the control.** Paired over seeds, the deficit of `K=1` is +0.06 pp with a 95% interval of [−0.52, +0.63], and `K=1 L` is nominally 0.11 pp *above* the control. The intervals for all four K=1
variants contain zero. This is achieved with 0.38 spikes per neuron in the attention and identical parameter count to the control.

**Neither variant has a detectable effect.** Treating sign and learnable thresholds as a 2×2 factorial at each K and testing the main effects paired over seeds:

<div align="center">

| K | effect of `±` | effect of `L` |
|---|---|---|
| 1 | −0.42 pp [−0.97, +0.13] | +0.07 pp [−0.24, +0.37] |
| 2 | +0.13 pp [−0.40, +0.67] | −0.07 pp [−0.32, +0.18] |
| 3 | +0.33 pp [−0.25, +0.92] | +0.13 pp [−0.12, +0.38] |

</div>

All six intervals contain zero, and they are narrow enough that this is an absence of effect rather than a lack of power. The sign additionally costs about 70% more spikes.

**More levels do not help here, and slightly hurt.** `K=2` and `K=3` sit 0.81 pp [+0.17, +1.45] and 0.99 pp [+0.42, +1.55] below the control while consuming two and three times the spikes. The likely reason is the window ceiling
described above: at `s = 0.25` the resolved range only widens from 0.250 to 0.438 between K=1 and K=3, so saturation falls from 40% to 33% while the spike count triples. `qk_scale` is held fixed across K in this sweep, which is the
appropriate choice for an ablation but biases it against the higher-resolution configurations; a per-K calibration is listed under future work.

### JetNet: gluon vs light-quark tagging

Real jets from 13 TeV proton–proton collisions, 30 particles each, as `(η_rel, φ_rel, p_T_rel)` point clouds, structurally identical to the synthetic hits, so the model runs unchanged. Features are standardised on training statistics
computed over valid particles only. 10 seeds, 15 epochs, 60k train / 20k test.

![SSA comparison](figures/ssa_jetnet_colab.png)

<div align="center">

| Configuration | Accuracy | AUC | rej@50% | Params | Attn spikes |
|---|---|---|---|---|---|
| softmax_bn + GELU | **79.02% ± 0.39** | **0.8719 ± 0.0018** | 23.9 | 103,298 | — |
| FS-SSA K=1 | 78.26% ± 0.38 | 0.8653 ± 0.0014 | 23.6 | 103,298 | 0.39 |
| FS-SSA K=1 ± | 78.66% ± 0.25 | 0.8667 ± 0.0017 | 23.2 | 103,298 | 0.66 |
| FS-SSA K=1 L | 78.43% ± 0.41 | 0.8657 ± 0.0020 | 23.8 | 105,986 | 0.39 |
| FS-SSA K=1 ± L | 78.73% ± 0.30 | 0.8672 ± 0.0016 | 23.6 | 106,754 | 0.66 |
| FS-SSA K=2 | 78.68% ± 0.45 | 0.8686 ± 0.0028 | 23.5 | 103,298 | 0.75 |
| FS-SSA K=2 ± | 78.72% ± 0.38 | 0.8675 ± 0.0024 | 23.5 | 103,298 | 1.32 |
| **FS-SSA K=2 L** | 78.76% ± 0.38 | **0.8695 ± 0.0020** | 23.7 | 108,674 | 0.75 |
| FS-SSA K=2 ± L | 78.74% ± 0.40 | 0.8673 ± 0.0025 | 23.3 | 110,210 | 1.31 |
| FS-SSA K=3 | 78.72% ± 0.39 | 0.8692 ± 0.0026 | 23.3 | 103,298 | 1.13 |
| FS-SSA K=3 ± | 78.87% ± 0.33 | 0.8684 ± 0.0022 | 23.2 | 103,298 | 1.98 |
| FS-SSA K=3 L | 78.76% ± 0.46 | 0.8694 ± 0.0030 | 23.4 | 111,362 | 1.13 |
| **FS-SSA K=3 ± L** | **78.94% ± 0.38** | 0.8689 ± 0.0025 | 23.7 | 113,666 | 1.97 |

</div>

Any model must clearly beat a trivial baseline that sees only the number of particles in the jet: gluon jets have higher multiplicity than quark jets, and that count alone reaches **67.7%** on this test split (majority class 50.8%). All
configurations do, by more than ten points.

**On measured data the deficit is real but small, and it shrinks with K.** Paired over seeds, `K=1` is 0.77 pp [+0.46, +1.07] and 0.0066 AUC [+0.0048, +0.0083] below the control; `K=3` is 0.30 pp [+0.04, +0.56] and 0.0027 AUC
[+0.0008, +0.0046] below it. The direction is opposite to the synthetic task, where higher K was slightly worse. This is because these are two different problems (30 and 120 tokens respectively), and that `qk_scale` is held at a
single value across both.

**The sign helps only at K=1.** In the 2×2 factorial its main effect is +0.36 pp [+0.18, +0.53] at K=1, then +0.00 pp at K=2 and +0.16 pp [−0.01, +0.33] at K=3, all on accuracy. On AUC the picture does not follow: the highest AUC at K=2
and K=3 belongs to the unsigned configurations. Learnable thresholds are not significant at any K on either metric. The reading consistent with both datasets is that the sign supplies resolution which becomes redundant once the neuron has
more than one level, at a cost of roughly twice the spikes.

## Diagnostics

Beyond accuracy, three properties were measured directly on the blocks. These are initialisation-time or architectural measurements and were not affected by the evaluation correction described above.

### Cost per attention layer

Both mechanisms are O(T² d) so the asymptotic complexity is identical. What differs is the *kind* of operation and how many actually run:

<div align="center">

| | MAC | AC | EXP | effective ops |
|---|---|---|---|---|
| softmax + GELU | 1,874,048 | 0 | 58,564 | 100% |
| FS-SSA K=1 | **0** | 1,874,048 | **0** | **25.1%** |
| FS-SSA K=4 | 0 | 1,874,048 | 0 | 29.9% |

</div>

A spiking product is a select rather than a multiply, and with `d_k = 2^-k` it is a shift-and-add. No transcendental functions are evaluated at all, and the sparsity of Q and K removes three quarters of the remaining work. K barely
affects the cost.

### Stability in sequence length and depth

<div align="center">

| T | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|
| output std | 0.177 | 0.176 | 0.174 | 0.175 |

</div>

The output scale is flat over two orders of magnitude in sequence length so the direct check that row normalisation works, since without it the output would grow linearly with T.

<div align="center">

| Blocks | 1 | 4 | 12 | 24 |
|---|---|---|---|---|
| FS layers | 4 | 16 | 48 | 96 |
| gradient at embedding | 0.318 | 0.365 | 0.395 | 0.406 |

</div>

The gradient does **not** vanish through 96 stacked spiking layers; if anything it grows slightly. Activity is uniform across depth (0.82–0.85 spikes/neuron, ~54% silent at every layer): no layer switches off, none saturates.

### Padding

Padded positions leak into the output in `train()` mode, because `BatchNorm1d` over `(B, C, T)` pools statistics over time as well as batch. In `eval()` the invariance is exact. This is a property of the normalisation choice, shared by
the SSA branch and by the matched control, and it is asserted in both directions in `Attention.py`.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.neurons          # FS variants: parity, antisymmetry, gradients
python Attention.py            # output scale, padding invariance, stability in T and K
python model.py                # ablation, depth stability, layer-by-layer diagnostics
python Experiment.py           # synthetic sweep (reduce N_TRAIN/EPOCHS on CPU)
```

The full sweeps are meant for Colab with a GPU: `Experiment_Colab.py` and `Jetnet_Experiment_Colab.py` are self-contained and run there directly. The synthetic sweep is 13 configurations × 19 seeds; the JetNet one trains on a 60k-jet
subsample.

`results/*.json` are included, so every figure and table above can be regenerated without retraining. Each file carries both readings  `accuracy` from exact BatchNorm statistics and `accuracy_ema` from the running average, so the size of the correction can be inspected per configuration and per seed.

## Repository structure

```
src/
├── neurons.py               # surrogate, FSNeuron, signed and learnable variants,
│                            # spike accounting
└── data/
    └── temporal_rings.py    # synthetic (x, y, t) hit sequences
Attention.py                 # softmax, BatchNorm-softmax, SpikingSelfAttention
model.py                     # switchable block and transformer
Experiment.py                # synthetic sweep
Experiment_Colab.py          # same sweep, self-contained for Colab
Jetnet_Experiment_Colab.py   # JetNet sweep, self-contained for Colab
results/                     # raw results (JSON)
figures/                     # generated plots
```

Dependencies run in one direction only: `Attention.py` imports from `src/neurons.py`, `model.py` from both, so each module can be tested on its own. Each module's `__main__` block asserts the invariants that the bugs listed above
violated, so a regression fails at import time rather than after a sweep.

## Limitations and future work

- **`qk_scale` is held fixed across K.** This is correct for an ablation but biases it, because the resolved input window has a ceiling at twice the threshold scale: raising K refines the quantisation step without widening the range. A *per-K calibration* selected on a validation split is the natural next measurement, and is the most likely explanation for the two datasets disagreeing about whether more levels help.

- **The variants carry a parameter overhead.** The learnable ladders add 2.5–9.5% parameters over the matched control, more when combined with the ON/OFF populations. The comparison is otherwise paired: same BatchNorm, same initialisation
  per seed, same data.

- **One architecture size.** All results use `d_model = 64`, `depth = 2`.

- **Spike and operation counts are proxies for energy**, not measurements. Unstructured sparsity translates into savings only on hardware built to exploit it.

- **Only Q, K and V are spiking.** The linear projections, the normalisations, the attention–value product and the classification head *remain* in floating point. Quantising the attention–value product is the substantive next step, since it
  requires re-encoding a real-valued attention matrix and is where a further loss is expected.

- **The surrogate gradient is triangular**, i.e. compactly supported: a neuron drifting more than `width` from its thresholds receives exactly zero gradient and cannot recover. Surrogates with infinite support (fast sigmoid, arctan) are a
  natural experiment.

- **BatchNorm over Q/K/V pools statistics over time**, which rules the current block out for autoregressive use: in a causal model the statistics at position *t* would include future tokens. LayerNorm or RMSNorm removes both that problem
  and the running-statistics discrepancy described above, at the cost of changing the matched control.

## References

- Stöckl & Maass, *Optimized spiking neurons can classify images with high accuracy through temporal coding with two spikes*, Nature Machine Intelligence 3, 230–238 (2021).

- Zhou et al., *Spikformer: When Spiking Neural Network Meets Transformer*, ICLR 2023.

- Kansal et al., *JetNet: A Python package for accessing open datasets and benchmarking machine learning methods in high energy physics*, JOSS 8(90), 5789 (2023).

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
