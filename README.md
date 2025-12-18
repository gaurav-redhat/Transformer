<p align="center">
  <img src="./images/hero-banner.svg" alt="The Complete Transformer Guide" width="100%">
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/🚀-Quick_Start-blue?style=for-the-badge" alt="Quick Start"></a>
  <a href="#-the-problem"><img src="https://img.shields.io/badge/⚠️-Problem-red?style=for-the-badge" alt="Problem"></a>
  <a href="#-the-solution"><img src="https://img.shields.io/badge/✅-Solution-green?style=for-the-badge" alt="Solution"></a>
  <a href="#-optimizations"><img src="https://img.shields.io/badge/⚡-Optimizations-orange?style=for-the-badge" alt="Optimizations"></a>
  <a href="#-2025-sota"><img src="https://img.shields.io/badge/🚀-2025_SOTA-purple?style=for-the-badge" alt="2025 SOTA"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Diagrams-41+-00BCD4?style=flat-square&logo=image" alt="Diagrams">
  <img src="https://img.shields.io/badge/Architectures-20+-4CAF50?style=flat-square&logo=buffer" alt="Architectures">
  <img src="https://img.shields.io/badge/Math-100%25-FF9800?style=flat-square&logo=mathworks" alt="Math">
  <img src="https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Years-2017--2025-9C27B0?style=flat-square&logo=calendar" alt="Years">
</p>

---

## 📍 Visual Learning Path

<p align="center">
  <img src="./images/visual-toc.svg" alt="Visual Table of Contents" width="100%">
</p>

<details>
<summary><b>📚 Detailed Table of Contents</b></summary>

1. [🚀 Quick Start](#-quick-start)
2. [⚠️ The Problem: Why We Needed Transformers](#-the-problem)
3. [✅ The Solution: Self-Attention](#-the-solution)
4. [📐 Step-by-Step Math](#-step-by-step-math)
5. [🔧 Key Components Deep Dive](#-key-components)
6. [📈 Evolution Timeline: 2017 → 2025](#-evolution-timeline)
7. [🔄 Architecture Comparisons](#-architecture-comparisons)
8. [💻 Complete Implementation](#-complete-implementation)
9. [⚡ Optimizations](#-optimizations)
10. [📊 Complexity: O(n²) → O(n)](#-complexity-reduction)
11. [🚀 2025 SOTA: Advanced Attention](#-2025-sota)
12. [📖 Further Reading](#-further-reading)

</details>

---

## 🚀 Quick Start

<table>
<tr>
<td width="50%">

### What You'll Learn

| Topic | Coverage |
|:------|:---------|
| 🧠 **Core Concepts** | Self-attention, Multi-head, QKV |
| 📐 **Mathematics** | Full derivations with examples |
| 🏗️ **Architectures** | 20+ transformer variants |
| ⚡ **Optimizations** | Flash Attention, KV Cache, etc. |
| 🔬 **2025 Research** | GLA, Mamba, KDA, Kimi Linear |

</td>
<td width="50%">

### Who Is This For?

- 🎓 **Students** learning deep learning
- 👨‍💻 **Engineers** implementing transformers
- 🔬 **Researchers** exploring new architectures
- 📊 **ML Practitioners** optimizing models

**Prerequisites**: Basic linear algebra & Python

</td>
</tr>
</table>

---

## ⚠️ The Problem

<p align="center">
  <img src="./images/rnn-problem.svg" alt="RNN Problems" width="100%">
</p>

### Why RNNs Failed for Long Sequences

<table>
<tr>
<td width="33%" align="center">

### 🐌 Sequential

```
Time: O(n)
Can't parallelize!
```

Each step waits for previous

</td>
<td width="33%" align="center">

### 📉 Vanishing Gradients

```
Step 1:  1.0
Step 50: 0.00001
```

Information gets lost

</td>
<td width="33%" align="center">

### 🔗 Long-Range

```
Token 1 ←❌→ Token 1000
```

Hard to connect distant tokens

</td>
</tr>
</table>

### The Math Problem

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_n} \cdot \prod_{t=2}^{n} W_{hh}^T \cdot \text{diag}(\tanh'(z_t))$$

**If** $\|W_{hh}\| < 1$ **→ Gradients vanish exponentially!**

---

## ✅ The Solution

<p align="center">
  <img src="./images/transformer-solution.svg" alt="Transformer Solution" width="100%">
</p>

### Self-Attention: Direct Connections

<p align="center">
  <img src="./images/attention-mechanism.svg" alt="Attention Mechanism" width="100%">
</p>

> **Key Insight**: Every token can directly attend to every other token!

### The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<table>
<tr>
<td width="33%" align="center">
<h4>Query (Q)</h4>
"What am I looking for?"
</td>
<td width="33%" align="center">
<h4>Key (K)</h4>
"What do I contain?"
</td>
<td width="33%" align="center">
<h4>Value (V)</h4>
"What do I offer?"
</td>
</tr>
</table>

---

## 📐 Step-by-Step Math

<p align="center">
  <img src="./images/attention-math.svg" alt="Attention Math" width="100%">
</p>

### 1️⃣ Create Q, K, V

```python
Q = X @ W_Q  # (n, d) @ (d, d_k) = (n, d_k)
K = X @ W_K  # (n, d) @ (d, d_k) = (n, d_k)  
V = X @ W_V  # (n, d) @ (d, d_v) = (n, d_v)
```

### 2️⃣ Compute Attention Scores

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$

### 3️⃣ Apply Softmax

$$\text{weights} = \text{softmax}(\text{scores})$$

### 4️⃣ Weighted Sum

$$\text{output} = \text{weights} \cdot V$$

---

## 🔧 Key Components

### Multi-Head Attention

<p align="center">
  <img src="./images/multi-head-attention.svg" alt="Multi-Head Attention" width="100%">
</p>

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

### Positional Encoding

<p align="center">
  <img src="./images/positional-encoding.svg" alt="Positional Encoding" width="100%">
</p>

<table>
<tr>
<td width="50%">

#### Sinusoidal (2017)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

</td>
<td width="50%">

#### RoPE (2023+)

<p align="center">
  <img src="./images/rope-position.svg" alt="RoPE" width="100%">
</p>

</td>
</tr>
</table>

### Feed-Forward Network

<p align="center">
  <img src="./images/feed-forward-network.svg" alt="FFN" width="100%">
</p>

### Normalization Evolution

<p align="center">
  <img src="./images/layer-norm-math.svg" alt="LayerNorm vs RMSNorm" width="100%">
</p>

### Activation Functions

<p align="center">
  <img src="./images/swiglu-activation.svg" alt="SwiGLU" width="100%">
</p>

---

## 📈 Evolution Timeline

<p align="center">
  <img src="./images/transformer-timeline.svg" alt="Transformer Timeline" width="100%">
</p>

### Year-by-Year Block Diagrams

<details>
<summary><b>🔍 2017: Original Transformer</b></summary>

<p align="center">
  <img src="./images/2017-original-transformer-block.svg" alt="2017 Transformer" width="100%">
</p>

</details>

<details>
<summary><b>🔍 2018: BERT & GPT</b></summary>

<p align="center">
  <img src="./images/2018-bert-gpt-blocks.svg" alt="2018 BERT GPT" width="100%">
</p>

</details>

<details>
<summary><b>🔍 2020: GPT-3 & Scaling</b></summary>

<p align="center">
  <img src="./images/2020-gpt3-block.svg" alt="2020 GPT-3" width="100%">
</p>

</details>

<details>
<summary><b>🔍 2023: LLaMA & Mistral</b></summary>

<p align="center">
  <img src="./images/2023-llama-mistral-block.svg" alt="2023 LLaMA Mistral" width="100%">
</p>

</details>

<details>
<summary><b>🔍 2024-2025: Modern Architectures</b></summary>

<p align="center">
  <img src="./images/2024-2025-modern-block.svg" alt="2024-2025 Modern" width="100%">
</p>

</details>

---

## 🔄 Architecture Comparisons

<p align="center">
  <img src="./images/all-transformers-comparison.svg" alt="All Transformers Comparison" width="100%">
</p>

### Encoder vs Decoder

<p align="center">
  <img src="./images/encoder-decoder.svg" alt="Encoder Decoder" width="100%">
</p>

### Mixture of Experts

<p align="center">
  <img src="./images/moe-architecture.svg" alt="MoE Architecture" width="100%">
</p>

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

---

## ⚡ Optimizations

<p align="center">
  <img src="./images/all-optimizations.svg" alt="All Optimizations Overview" width="100%">
</p>

### Flash Attention

<p align="center">
  <img src="./images/flash-attention.svg" alt="Flash Attention" width="100%">
</p>

### KV Cache

<p align="center">
  <img src="./images/kv-cache.svg" alt="KV Cache" width="100%">
</p>

### Quantization

<p align="center">
  <img src="./images/quantization.svg" alt="Quantization" width="100%">
</p>

### Speculative Decoding

<p align="center">
  <img src="./images/speculative-decoding.svg" alt="Speculative Decoding" width="100%">
</p>

### Gradient Checkpointing

<p align="center">
  <img src="./images/gradient-checkpointing.svg" alt="Gradient Checkpointing" width="100%">
</p>

---

## 📊 Complexity Reduction

<p align="center">
  <img src="./images/attention-complexity-comparison.svg" alt="Attention Complexity" width="100%">
</p>

### Sparse Attention Patterns

<p align="center">
  <img src="./images/sparse-attention.svg" alt="Sparse Attention" width="100%">
</p>

### Linear / Kernel Attention

<p align="center">
  <img src="./images/linear-attention.svg" alt="Linear Attention" width="100%">
</p>

**The Kernel Trick**:
$$\text{softmax}(QK^T) \approx \phi(Q)\phi(K)^T$$

Reorder: $(QK^T)V \rightarrow Q(K^TV)$ → **O(n²) to O(n)!**

### Low-Rank Attention (Linformer)

<p align="center">
  <img src="./images/low-rank-attention.svg" alt="Low-Rank Attention" width="100%">
</p>

### LSH / Hash Attention

<p align="center">
  <img src="./images/lsh-attention.svg" alt="LSH Attention" width="100%">
</p>

### Clustering Attention

<p align="center">
  <img src="./images/clustering-attention.svg" alt="Clustering Attention" width="100%">
</p>

---

## 🚀 2025 SOTA

<p align="center">
  <img src="./images/attention-evolution-2025.svg" alt="2025 SOTA" width="100%">
</p>

### Delta Rule & DeltaNet

<p align="center">
  <img src="./images/delta-rule-attention.svg" alt="Delta Rule" width="100%">
</p>

$$S_t = S_{t-1} + k_t \beta_t (v_t - S_{t-1}^T k_t)^T$$

**Key**: Subtract old association before adding new!

### Gated Linear Attention (GLA)

<p align="center">
  <img src="./images/gated-linear-attention.svg" alt="GLA" width="100%">
</p>

$$S_t = G_t \odot S_{t-1} + k_t \otimes v_t$$

### Multi-Head Latent Attention (MLA)

<p align="center">
  <img src="./images/mla-attention.svg" alt="MLA" width="100%">
</p>

**8× KV cache compression** from DeepSeek-V2!

### State Space Models (Mamba)

<p align="center">
  <img src="./images/state-space-models.svg" alt="Mamba" width="100%">
</p>

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

**O(n) time, O(1) state!**

### Hybrid Architectures (Kimi Linear)

<p align="center">
  <img src="./images/hybrid-attention.svg" alt="Hybrid Attention" width="100%">
</p>

From [Kimi Linear Paper (arXiv:2510.26692)](https://arxiv.org/pdf/2510.26692):

| Metric | Full MLA | Kimi Linear |
|:-------|:---------|:------------|
| MMLU-Pro | 49.2 | **51.0** (+1.8) |
| RULER@128K | 81.1 | **84.3** (+3.2) |
| Speed@1M | 1× | **6.3×** |
| KV Cache | 100% | **25%** |

> "For the first time, outperforms full attention under fair comparisons"

---

## 📊 Master Comparison Table

| Method | Time | Memory | Quality | Best For |
|:-------|:-----|:-------|:--------|:---------|
| Standard | O(n²d) | O(n²) | 100% | Short sequences |
| Flash Attention | O(n²d) | **O(n)** | 100% | Training |
| Sliding Window | O(nwd) | O(nw) | ~98% | Long docs |
| Linear/Kernel | O(nd²) | O(nd) | ~95% | Very long |
| Linformer | O(nkd) | O(nk) | ~96% | Fixed compression |
| LSH (Reformer) | O(n log n) | O(n log n) | ~97% | Sparse |
| GLA | O(nd²) | O(nd) | ~98% | Efficient |
| DeltaNet | O(nd²) | O(nd) | ~99% | Quality |
| Mamba | O(nd) | O(d) | ~97% | Speed |
| **KDA (Kimi)** | O(nd²) | O(nd) | **101%+** | **Everything!** |

---

## 📖 Further Reading

### Papers

| Year | Paper | Innovation |
|:-----|:------|:-----------|
| 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original Transformer |
| 2018 | [BERT](https://arxiv.org/abs/1810.04805) | Bidirectional pre-training |
| 2020 | [GPT-3](https://arxiv.org/abs/2005.14165) | Scale is all you need |
| 2022 | [Flash Attention](https://arxiv.org/abs/2205.14135) | IO-aware attention |
| 2023 | [LLaMA](https://arxiv.org/abs/2302.13971) | Efficient open models |
| 2023 | [Mamba](https://arxiv.org/abs/2312.00752) | Selective SSM |
| 2024 | [GLA](https://arxiv.org/abs/2312.06635) | Gated linear attention |
| 2025 | [Kimi Linear](https://arxiv.org/abs/2510.26692) | KDA + MLA hybrid |

### Code Resources

- 🔥 [Hugging Face Transformers](https://github.com/huggingface/transformers)
- ⚡ [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- 🐍 [Mamba](https://github.com/state-spaces/mamba)
- 🚀 [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)

---

<p align="center">

### 🌟 Star this repo if you found it helpful!

**Made with ❤️ for the ML community**

<a href="#-quick-start"><img src="https://img.shields.io/badge/↑_Back_to_Top-333?style=for-the-badge" alt="Back to Top"></a>

</p>

---

<p align="center">
  <sub>📝 Last updated: December 2025 | 📧 Contributions welcome!</sub>
</p>
