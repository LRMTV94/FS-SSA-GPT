# =====================================================================
#  FS-SSA, autoregressive: BPE GPT on TinyStories
#  Single-file Colab script. Runtime > Change runtime type > GPU
#
#  Byte-identical to the tiny-Shakespeare script except for the data section,
#  the model/budget constants, gradient accumulation in the training loop and
#  the output paths. Every class -- the FS neurons, both attentions, Block,
#  FSGPT -- is unchanged, so the two experiments can be read against each
#  other and a seed means the same thing in both.
#
#  Measures validation loss and perplexity for a causal spiking attention
#  against a matched softmax control, and samples text from each.
#
#  ---------------------------------------------------------------------
#  THREE THINGS DIFFER FROM THE CLASSIFIER, AND ALL THREE ARE FORCED
#
#  1) RMSNorm replaces BatchNorm on Q/K/V. BatchNorm1d over (B, C, T) pools
#     statistics over TIME as well as batch, so in a causal model the
#     statistics at position t would include future tokens: a direct leak.
#     RMSNorm normalises over the channel dimension only. As a side effect
#     it also removes the running-statistics discrepancy that dominated the
#     classifier results, since RMSNorm keeps no buffers.
#
#  2) The row normalisation becomes causal. Without a softmax the attention
#     rows do not sum to 1 and must be divided by the number of attended
#     keys; in a causal model position t attends to t+1 keys, not to a
#     constant. Dividing by a constant would crush the first positions.
#
#  3) qk_scale is re-measured, not inherited. The value 0.25 was calibrated
#     for a BatchNorm-ed input; the spread after RMSNorm is different. The
#     script probes the pre-activation std at init and derives the scale
#     from it. Recall that the resolved window is [0, s*(2 - 2^-(K-1))),
#     with a ceiling at 2s: raising K refines the step, never the range.
#
# =====================================================================

import os
import json
import math
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tiktoken
except ImportError:
    raise SystemExit("pip install tiktoken")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

DATA_URL   = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
DATA_PATH  = "TinyStories-train.txt"
TOKEN_PATH = "TinyStories-train-gpt2.bin"
MIN_BYTES  = 100 * 2**20

BLOCK      = 256         # context length
D_MODEL    = 384
N_HEADS    = 6
N_LAYER    = 12
MLP_RATIO  = 4
DROPOUT    = 0.0          # under one epoch of 450M tokens: nothing to regularise

MAX_ITERS      = 5000
EVAL_INTERVAL  = 250
EVAL_ITERS     = 40       # batches averaged per evaluation
MICRO_BATCH    = 16       # what goes on the GPU at once
GRAD_ACCUM     = 4        # effective batch = MICRO_BATCH * GRAD_ACCUM = 64
LR             = 1e-3
WARMUP         = 200
MIN_LR         = 1e-4
GRAD_CLIP      = 1.0
SEEDS          = [0, 1, 2]


WIDTH          = 1.0      # surrogate half-width
READOUT_SCALE  = 1.0      # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT  = 0.75     # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT = 1.0      # mlp_scale = this x measured std of the MLP pre-activation

#  (name, attention, activation, K, signed, learnable)
#   attention  -> softmax with causal mask, or SSA (softmax-free, FS-coded QKV)
#   signed     -> ON/OFF pair on Q and K, so they carry a sign
#   learnable  -> per-channel learnable threshold ladder, attention AND MLP

CONFIGS = [
    #("softmax + gelu", "softmax", "gelu", 2, False, False),   # matched control
    #("ssa K=2",        "ssa",     "fs",   2, False, False),   # spiking attention
    #("ssa K=2 +/-",    "ssa",     "fs",   2, True,  False),   # + sign, restores suppression
    ("ssa K=2 +/- L",  "ssa",     "fs",   2, True,  True),    # + learnable thresholds
]

TAG          = "tinystories"
CKPT_PATH    = f"fsssa_{TAG}_state.json"
RESULTS_PATH = f"results_fsssa_{TAG}.json"
PLOT_PATH    = f"fsssa_{TAG}_curves.png"
SAMPLES_PATH = f"fsssa_{TAG}_samples.txt"

GEN_TOKENS      = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 200


# =====================================================================
#                              DATA
# =====================================================================

tokenizer = tiktoken.get_encoding("gpt2")
VOCAB = tokenizer.n_vocab
EOT = tokenizer.eot_token
decode = lambda ids: tokenizer.decode([int(i) for i in ids])


def build_tokens():
    '''Download, tokenise in chunks, cache to disk as int32'''

    if os.path.exists(TOKEN_PATH):
        arr = np.fromfile(TOKEN_PATH, dtype=np.int32)
        print(f"tokens loaded from cache: {len(arr):,}")
        return torch.from_numpy(arr.astype(np.int64))

    if not os.path.exists(DATA_PATH):
        print(f"downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    size = os.path.getsize(DATA_PATH)
    if size < MIN_BYTES:
        head = open(DATA_PATH, "rb").read(200)
        raise SystemExit(
            f"{DATA_PATH} is only {size/2**20:.2f} MB -- almost certainly the\n"
            f"git-LFS pointer rather than the file. First bytes:\n  {head!r}\n"
            f"Check that the URL uses resolve/main and not raw/main.")
    print(f"{DATA_PATH}: {size/2**30:.2f} GB")

    ids, n_chunk = [], 0
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= 200_000:                       # ~50 MB of text
                ids.append(np.asarray(tokenizer.encode_ordinary("".join(buf)),
                                      dtype=np.int32))
                buf, n_chunk = [], n_chunk + 1
                print(f"  chunk {n_chunk}: {sum(len(a) for a in ids):,} tokens",
                      flush=True)
        if buf:
            ids.append(np.asarray(tokenizer.encode_ordinary("".join(buf)),
                                  dtype=np.int32))

    arr = np.concatenate(ids)
    del ids
    arr.tofile(TOKEN_PATH)
    print(f"tokenised: {len(arr):,} tokens, cached to {TOKEN_PATH}")
    return torch.from_numpy(arr.astype(np.int64))


data = build_tokens()
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"TinyStories (GPT-2 BPE): {len(data):,} tokens | vocab {VOCAB:,} | train {len(train_data):,} | val {len(val_data):,}")
print(f"  {MAX_ITERS} x {MICRO_BATCH*GRAD_ACCUM} x {BLOCK} = {MAX_ITERS*MICRO_BATCH*GRAD_ACCUM*BLOCK/1e6:.0f}M tokens seen = {MAX_ITERS*MICRO_BATCH*GRAD_ACCUM*BLOCK/len(train_data):.2f} epochs")


def get_batch(split, generator=None):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (MICRO_BATCH,), generator=generator)
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return x.to(device), y.to(device)


# =====================================================================
#                    FS NEURONS + SPIKE ACCOUNTING
# =====================================================================

_SPIKE_ON = False          # counting is off during training


def set_spike_counting(on):
    global _SPIKE_ON
    _SPIKE_ON = on


def reset_spike_stats(model):
    for m in model.modules():
        if hasattr(m, "spike_sum"):
            m.spike_sum, m.spike_n = 0.0, 0


def _record(module, spike_count):
    if not _SPIKE_ON:
        return
    sc = spike_count.detach()
    module.spike_sum += float(sc.sum().item())
    module.spike_n += int(sc.numel())


class TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.clamp(1.0 - x.abs() / ctx.width, min=0.0), None

spike = TriangularSpike.apply


def fs_window(K, s):
    ''' Input range the neuron resolves: [0, window). Ceiling is 2s '''
    return s * (2 - 2.0 ** -(K - 1))


class FSNeuron(nn.Module):

    ''' T, h scaled by threshold_scale; d by readout_scale. The two are independent: d has no influence on which spikes fire'''

    def __init__(self, K, width, threshold_scale, readout_scale):
        super().__init__()
        self.K = K
        self.width = width
        self.threshold_scale = threshold_scale
        self.readout_scale = readout_scale
        self.spike_sum = 0.0
        self.spike_n = 0

        geom = 2.0 ** -(torch.arange(K, dtype=torch.float32))
        self.register_buffer("T", threshold_scale * geom.clone())
        self.register_buffer("h", threshold_scale * geom.clone())
        self.register_buffer("d", readout_scale * geom.clone())

    def forward(self, x):
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - self.T[i], self.width)
            out = out + s * self.d[i]
            v = v - s * self.h[i]
            cnt = cnt + s
        _record(self, cnt)
        return out


class LearnableFSNeuron(nn.Module):

    ''' FS neuron with learnable thresholds, optionally one ladder per channel'''

    def __init__(self, K, surrogate_width, threshold_scale, readout_scale, per_channel, n_channels):
        super().__init__()
        self.K = K
        self.width = surrogate_width
        self.per_channel = per_channel

        self.threshold_scale = threshold_scale
        self.readout_scale = readout_scale
        self.spike_sum = 0.0
        self.spike_n = 0

        geom = 2.0 ** -(torch.arange(K, dtype=torch.float32))
        raw_thr = self._invert(threshold_scale * geom)                          # T, h
        raw_out = self._invert(readout_scale * geom)                            # d

        if per_channel:
            raw_thr = raw_thr[:, None].repeat(1, n_channels)
            raw_out = raw_out[:, None].repeat(1, n_channels)

        self.raw_T = nn.Parameter(raw_thr.clone())
        self.raw_h = nn.Parameter(raw_thr.clone())
        self.raw_d = nn.Parameter(raw_out.clone())

    @staticmethod
    def _invert(values):
        increments = torch.cat([values[:-1] - values[1:], values[-1:]])
        return torch.log(torch.expm1(increments.clamp(min=1e-6)))

    @staticmethod
    def _ladder(raw):
        inc = F.softplus(raw)
        return inc.flip(0).cumsum(0).flip(0)

    def thresholds(self):
        return (self._ladder(self.raw_T), self._ladder(self.raw_d), self._ladder(self.raw_h))

    def forward(self, x):

        T, d, h = self.thresholds()
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - T[i], self.width)
            out = out + s * d[i]
            v = v - s * h[i]
            cnt = cnt + s
        _record(self, cnt)

        return out

    def ladder_stats(self):

        T, _, _ = self.thresholds()
        if self.per_channel:
            return {"T_mean": T.mean(dim=1).tolist(), "T_std": T.std(dim=1).tolist()}
        return {"T": T.tolist()}


def make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):

    ''' Single place where the FS variant is chosen. No bypasses '''

    if learnable:
        return LearnableFSNeuron(K, width, threshold_scale, readout_scale, per_channel, n_channels)
    return FSNeuron(K, width, threshold_scale, readout_scale)


class SignedFSNeuron(nn.Module):

    ''' ON/OFF pair: out = FS(x) - FS(-x) '''

    def __init__(self, K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):
        super().__init__()
        self.K = K
        self.width = width
        self.fs_on = make_fs(K, width, threshold_scale, readout_scale, learnable,
                             per_channel, n_channels)
        self.fs_off = make_fs(K, width, threshold_scale, readout_scale, learnable,
                              per_channel, n_channels)

    def forward(self, x):
        return self.fs_on(x) - self.fs_off(-x)


def make_activation(kind, K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):

    if kind == "gelu":
        return nn.GELU()
    if kind == "fs":
        return make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels)
    raise ValueError(f"unknown activation: {kind}")


# =====================================================================
#                              MODEL
# =====================================================================

class RMSNorm(nn.Module):

    '''Normalises over channels only'''

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):

    '''Standard softmax attention with a causal mask'''

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads

        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK, dtype=torch.bool)))

    def forward(self, x):

        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = h(q), h(k), h(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(dim=-1)
        att = self.attn_drop(att)

        return self.resid_drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, C)))


class CausalSpikingSelfAttention(nn.Module):

    '''Softmax-free causal attention with FS-coded Q, K, V.

    Q and K are non-negative (unless signed), so every logit is >= 0 and the
    causal mask is a multiplication rather than a -inf fill. Rows are divided
    by the number of keys each position actually attends to, which grows with
    t: dividing by a constant would crush the beginning of the sequence'''

    def __init__(self, d_model, n_heads, K, width, qk_scale, readout_scale,
                 signed, learnable, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.signed = signed
        self.learnable = learnable

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.norm_v = RMSNorm(d_model)
        self.resid_drop = nn.Dropout(dropout)

        def enc():
            if signed:
                return SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)
            return make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.fs_q = enc()
        self.fs_k = enc()
        self.fs_v = make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK)))
        self.register_buffer("n_keys", torch.arange(1, BLOCK + 1, dtype=torch.float32))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.fs_q(self.norm_q(q))
        k = self.fs_k(self.norm_k(k))
        v = self.fs_v(self.norm_v(v))

        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = h(q), h(k), h(v)

        att = (q @ k.transpose(-2, -1)) * self.attn_scale
        att = att * self.mask[:T, :T]                      # causal
        att = att / self.n_keys[:T][None, None, :, None]   # causal row mean
        return self.resid_drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attention, activation, K, width, qk_scale, mlp_scale, readout_scale, signed, learnable, dropout):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        if attention == "softmax":
            self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        elif attention == "ssa":
            self.attn = CausalSpikingSelfAttention(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout)
        else:
            raise ValueError(attention)

        self.norm2 = RMSNorm(d_model)
        hidden = d_model * MLP_RATIO

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            make_activation(activation, K, width, mlp_scale, readout_scale,
                            learnable, True, hidden),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FSGPT(nn.Module):

    def __init__(self, attention, activation, K, signed, learnable, qk_scale, mlp_scale):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(BLOCK, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_HEADS, attention, activation, K, WIDTH, qk_scale, mlp_scale, READOUT_SCALE, signed, learnable, DROPOUT)
            for _ in range(N_LAYER)])

        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def spike_stats(self):
    
        '''Spikes per encoding site, per token, per channel'''
        
        leaves = lambda m: [x for x in m.modules()
                            if isinstance(x, (FSNeuron, LearnableFSNeuron))]

        def rate(sites):
            tot, cnt = 0.0, 0
            for s in sites:
                ls = leaves(s)
                if not ls or ls[0].spike_n == 0:
                    continue
                tot += sum(x.spike_sum for x in ls)
                cnt += ls[0].spike_n
            return tot / cnt if cnt else float("nan")

        attn, mlp = [], []
        for blk in self.blocks:
            attn += [getattr(blk.attn, nm) for nm in ("fs_q", "fs_k", "fs_v")
                     if getattr(blk.attn, nm, None) is not None]
            if leaves(blk.mlp[1]):
                mlp.append(blk.mlp[1])
        return {"attention": rate(attn), "mlp": rate(mlp)}


# =====================================================================
#                 THRESHOLD CALIBRATION AND CHECKS
# =====================================================================

@torch.no_grad()
def measure_scales():

    '''Probe the spread the FS neurons actually see, at initialisation'''

    torch.manual_seed(0)
    m = FSGPT("ssa", "fs", 2, False, False, 0.25, 1.0).to(device)
    qkv, pre = [], []

    hq = m.blocks[0].attn.norm_q.register_forward_hook(lambda mo, i, o: qkv.append(o.std().item()))
    hm = m.blocks[0].mlp[0].register_forward_hook(lambda mo, i, o: pre.append(o.std().item()))
    m.eval()

    for _ in range(4):
        m(get_batch("train")[0])
    hq.remove()
    hm.remove()

    del m
    return float(sum(qkv) / len(qkv)), float(sum(pre) / len(pre))


SIGMA_QK, SIGMA_MLP = measure_scales()
QK_SCALE = QK_SIGMA_MULT * SIGMA_QK
MLP_SCALE = MLP_SIGMA_MULT * SIGMA_MLP

print(f"\nmeasured std  Q/K/V after RMSNorm {SIGMA_QK:.4f} | MLP pre-activation {SIGMA_MLP:.4f}")
print(f"derived       qk_scale {QK_SCALE:.4f} ({QK_SIGMA_MULT:g} sigma) | mlp_scale {MLP_SCALE:.4f} ({MLP_SIGMA_MULT:g} sigma)")


def sanity_checks():

    print("\n--- FS sanity checks ---")
    x = torch.randn(4000, 8)

    for k in (1, 2, 3):

        a = make_fs(k, WIDTH, QK_SCALE, READOUT_SCALE, False, True, 8)
        b = make_fs(k, WIDTH, QK_SCALE, READOUT_SCALE, True, True, 8)
        err = (a(x) - b(x)).abs().max().item()
        assert err < 1e-5, f"K={k}: learnable does not start from fixed ({err:.2e})"

    print(f"  learnable == fixed at init : OK (max err {err:.2e})")

    hi = fs_window(2, QK_SCALE)
    z = hi * torch.rand(4000, 8)
    u = make_fs(2, WIDTH, QK_SCALE, QK_SCALE, False, True, 8)
    c = make_fs(2, WIDTH, QK_SCALE, READOUT_SCALE, False, True, 8)
    assert (c(z) - (READOUT_SCALE / QK_SCALE) * u(z)).abs().max() < 1e-5, "r is not a pure gain"
    print(f"  r is a pure gain           : OK ({READOUT_SCALE / QK_SCALE:.2f}x)")

    n_pl = sum(isinstance(m, SignedFSNeuron)
               for m in FSGPT("ssa", "fs", 2, False, False, QK_SCALE, MLP_SCALE).modules())
    n_sg = sum(isinstance(m, SignedFSNeuron)
               for m in FSGPT("ssa", "fs", 2, True, False, QK_SCALE, MLP_SCALE).modules())
    assert n_pl == 0 and n_sg > 0, "`signed` is not reaching the attention"
    print(f"  signed reaches attention   : OK ({n_sg} ON/OFF pairs)")

    # causality: a change at position t must not move the output before t
    torch.manual_seed(0)
    m = FSGPT("ssa", "fs", 2, False, False, QK_SCALE, MLP_SCALE).to(device).eval()
    with torch.no_grad():
        a = torch.randint(0, VOCAB, (2, 64), device=device)
        b = a.clone(); b[:, 32:] = torch.randint(0, VOCAB, (2, 32), device=device)
        d = (m(a)[0][:, :32] - m(b)[0][:, :32]).abs().max().item()
    assert d < 1e-4, f"causality violated: {d:.2e}"
    print(f"  causal mask                : OK (max leak {d:.2e})")

    x = torch.randn(4000, 8)
    print(f"  window at s={QK_SCALE:.3f} (K=2)   : [0, {hi:.3f}) saturated {(x > hi).float().mean().item()*100:.1f}%")
    print("--- end checks ---\n")


sanity_checks()


# =====================================================================
#                          TRAIN / EVAL
# =====================================================================

def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def run(name, attention, activation, K, signed, learnable, seed):
    torch.manual_seed(seed)
    model = FSGPT(attention, activation, K, signed, learnable, QK_SCALE, MLP_SCALE).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.99))

    hist = []
    best = {"val": float("inf"), "iter": -1, "train": float("nan")}
    t0 = time.time()
    model.train()

    for it in range(MAX_ITERS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            L = estimate_loss(model)
            hist.append({"iter": it, "train": L["train"], "val": L["val"]})
            if L["val"] < best["val"]:
                best = {"val": L["val"], "train": L["train"], "iter": it}
            mem = (f" | {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
                   if device == 'cuda' else "")
            print(f"    {name} s{seed} it {it:>5} | train {L['train']:.4f} "
                  f"| val {L['val']:.4f} (ppl {math.exp(L['val']):.2f}) "
                  f"| best {best['val']:.4f} @ {best['iter']} "
                  f"| {(time.time()-t0)/60:.0f} min{mem}", flush=True)

        if it == MAX_ITERS:
            break

        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = get_batch("train")
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    # spike rate, measured on the validation pass only
    set_spike_counting(True)
    reset_spike_stats(model)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(get_batch("val")[0])
    st = model.spike_stats()
    set_spike_counting(False)

    # learned ladders, if any: did the thresholds actually move?
    ladders = [m.ladder_stats() for m in model.modules()
               if isinstance(m, LearnableFSNeuron)][:3]

    # a sample, for qualitative inspection only
    model.eval()
    ctx = torch.full((1, 1), EOT, dtype=torch.long, device=device)
    sample = decode(model.generate(ctx, GEN_TOKENS, GEN_TEMPERATURE, GEN_TOP_K)[0].tolist())

    peak = torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else float("nan")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    return {"name": name, "seed": seed, "params": n_par, "peak_gb": peak,
            "best_val": best["val"], "best_iter": best["iter"],
            "best_train": best["train"], "final_val": hist[-1]["val"],
            "ppl": math.exp(best["val"]), "final_ppl": math.exp(hist[-1]["val"]),
            "spk_attn": st["attention"], "spk_mlp": st["mlp"],
            "minutes": (time.time() - t0) / 60, "history": hist,
            "ladders": ladders, "sample": sample}


# =====================================================================
#                             SWEEP
# =====================================================================

# The state file carries a fingerprint of the setup. Without it, resuming a
# run from a DIFFERENT experiment whose configuration names happen to match
# would skip every configuration and then print the old numbers under the new
# heading, silently. The measured threshold scales are deliberately NOT part
# of it: they jitter in the last digits, which would block legitimate resumes.
FINGERPRINT = {"data": TAG, "vocab": VOCAB, "block": BLOCK, "d_model": D_MODEL,
               "n_layer": N_LAYER, "n_heads": N_HEADS, "iters": MAX_ITERS,
               "eff_batch": MICRO_BATCH * GRAD_ACCUM, "lr": LR, "width": WIDTH,
               "qk_sigma_mult": QK_SIGMA_MULT, "mlp_sigma_mult": MLP_SIGMA_MULT,
               "readout_scale": READOUT_SCALE}
MEASURED = {"sigma_qk": SIGMA_QK, "sigma_mlp": SIGMA_MLP,
            "qk_scale": QK_SCALE, "mlp_scale": MLP_SCALE}

done = {}
if os.path.exists(CKPT_PATH):
    state = json.load(open(CKPT_PATH))
    if state.get("fingerprint") != FINGERPRINT:
        raise SystemExit(
            f"{CKPT_PATH} was written with a different setup and cannot be resumed.\n"
            f"  on disk: {state.get('fingerprint')}\n"
            f"  now    : {FINGERPRINT}\n"
            f"Move it aside or change TAG.")
    done = {f"{r['name']}|{r['seed']}": r for r in state["runs"]}
    print(f"resuming: {len(done)} runs already done")


def save():
    json.dump({"fingerprint": FINGERPRINT, "measured": MEASURED,
               "runs": list(done.values())}, open(CKPT_PATH, "w"), indent=1)

print("=" * 78)
print(f"TinyStories, GPT-2 BPE | {len(CONFIGS)} configs x {len(SEEDS)} seeds "
      f"| {MAX_ITERS} iters | block {BLOCK}, d_model {D_MODEL}, {N_LAYER} layers")
print(f"batch {MICRO_BATCH} x {GRAD_ACCUM} accum = {MICRO_BATCH*GRAD_ACCUM} effective")
print("=" * 78)

for name, attn, act, K, sg, ln in CONFIGS:
    for seed in SEEDS:
        key = f"{name}|{seed}"
        if key in done:
            print(f"  skip {key} (already done)")
            continue
        print(f"\n--- {name}, seed {seed} ---", flush=True)
        done[key] = run(name, attn, act, K, sg, ln, seed)
        save()

R = list(done.values())
json.dump({"fingerprint": FINGERPRINT, "measured": MEASURED, "runs": R},
          open(RESULTS_PATH, "w"), indent=1)


# =====================================================================
#                            SUMMARY
# =====================================================================

def agg(name, key):
    v = [r[key] for r in R if r["name"] == name]
    m = sum(v) / len(v)
    s = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
    return m, s


print("\n" + "=" * 90)
print(f"SUMMARY — best validation loss over {len(SEEDS)} seeds")
print("  (under one epoch of a 450M-token corpus: nothing can be memorised, so")
print("   `best` and `final` should nearly coincide)")
print("=" * 90)
print(f"{'config':>18} | {'best val loss':>17} | {'ppl':>7} | {'@iter':>6} | "
      f"{'final val':>9} | {'params':>9} | {'attn spk':>8} | {'min':>5}")
print("-" * 90)
for name, *_ in CONFIGS:
    bl, bs = agg(name, "best_val")
    it, _ = agg(name, "best_iter")
    fv, _ = agg(name, "final_val")
    sa, _ = agg(name, "spk_attn")
    mn, _ = agg(name, "minutes")
    par = [r["params"] for r in R if r["name"] == name][0]
    print(f"{name:>18} | {bl:.4f} +/- {bs:.4f} | {math.exp(bl):7.2f} | {it:6.0f} | "
          f"{fv:9.4f} | {par:>9,} | "
          f"{('     --' if math.isnan(sa) else f'{sa:7.3f}')} | {mn:5.1f}")

ref = CONFIGS[0][0]
rb, _ = agg(ref, "best_val")
print(f"\nDeficit vs {ref} (best val loss {rb:.4f}, ppl {math.exp(rb):.2f}), paired by seed:")
for name, *_ in CONFIGS[1:]:
    d = [b["best_val"] - a["best_val"]
         for a in R if a["name"] == ref
         for b in R if b["name"] == name and b["seed"] == a["seed"]]
    m = sum(d) / len(d)
    print(f"  {name:>18}: {m:+.4f} nats | ppl {math.exp(rb + m):.2f} vs {math.exp(rb):.2f}"
          f"   (n={len(d)}, individual: {', '.join(f'{x:+.4f}' for x in d)})")

if len(SEEDS) < 3:
    print(f"\nWith {len(SEEDS)} seed(s) these differences carry no usable interval: read")
    print("them as a direction to confirm, not as a measurement.")


# =====================================================================
#                        CURVES AND SAMPLES
# =====================================================================

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
cols = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
for (name, *_), c in zip(CONFIGS, cols):
    runs = [r for r in R if r["name"] == name]
    if not runs:
        continue
    its = [h["iter"] for h in runs[0]["history"]]
    val = [sum(r["history"][i]["val"] for r in runs) / len(runs) for i in range(len(its))]
    trn = [sum(r["history"][i]["train"] for r in runs) / len(runs) for i in range(len(its))]
    ax[0].plot(its, val, color=c, label=name)
    ax[0].plot(its, trn, color=c, ls=":", alpha=0.5)
    b = min(range(len(val)), key=lambda i: val[i])
    ax[0].scatter([its[b]], [val[b]], color=c, s=40, zorder=3)
    sa, _ = agg(name, "spk_attn")
    if not math.isnan(sa):
        bl, bs = agg(name, "best_val")
        ax[1].errorbar([sa], [bl], yerr=[bs], fmt='o', capsize=4, color=c,
                       markersize=9, label=name)

ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss")
ax[0].set_title("solid = validation, dotted = train, dot = best")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

rb, rs = agg(ref, "best_val")
ax[1].axhline(rb, color='tab:green', ls='--', lw=1.5, label=f'{ref} (control)')
ax[1].axhspan(rb - rs, rb + rs, color='tab:green', alpha=0.12)
ax[1].set_xlabel("spikes per neuron per token (attention)")
ax[1].set_ylabel("best validation loss")
ax[1].set_title("Loss vs spiking activity")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")

with open(SAMPLES_PATH, "w") as f:
    for r in R:
        f.write(f"{'='*70}\n{r['name']} | seed {r['seed']} | "
                f"best val {r['best_val']:.4f} (ppl {r['ppl']:.2f}) @ iter {r['best_iter']}\n"
                f"NOTE: sampled from the FINAL weights, not from the best checkpoint.\n"
                f"{'='*70}\n{r['sample']}\n\n")

print(f"\nSaved: {RESULTS_PATH}, {PLOT_PATH}, {SAMPLES_PATH}")
print("\n--- one sample per configuration (seed 0) ---")
for name, *_ in CONFIGS:
    s = [r for r in R if r["name"] == name and r["seed"] == SEEDS[0]]
    if s:
        print(f"\n=== {name} ===\n{s[0]['sample'][:400]}")
