import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timm import create_model, list_models
from types import SimpleNamespace
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
# Updated imports for torch.amp (replaces torch.cuda.amp)
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm.auto import tqdm
import gc
import json
import os

# Set up environment for Groq API
# Make sure GROQ_API_KEY is set in your environment
if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY environment variable not set!")
    print("Please set it with: export GROQ_API_KEY='your-api-key-here'")

import pandas as pd
df = pd.read_csv("./Dataset/reports.csv")
dataFrame = df[["image_name","findings"]]
dataFrame.rename(columns = {'image_name':'image','findings':'caption'}, inplace = True)
from pathlib import Path
base_path = Path('./Dataset/images/')
dataFrame['image'] = dataFrame['image'].apply(
    lambda x: str(base_path / x )
)


sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ToTensorV2()
])

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token

def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    image = torch.stack(image,dim=0)
    input_ids = tokenizer.pad(
        {'input_ids':input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    labels = tokenizer.pad(
        {'input_ids':labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    mask = (input_ids!=tokenizer.pad_token_id).long()
    labels[mask==0]=-100
    return image, input_ids, labels



from dataclasses import dataclass
import pennylane as qml  # pip install pennylane
from pennylane.templates import StronglyEntanglingLayers  # template alias

# qag_gpt2_integration.py

from dataclasses import dataclass
from typing import Optional, Tuple

# Optional import for QuantumAttentionGate internals
# If pennylane isn't installed and you don't pass qcfg, NoOpQuantumGate will be used.
try:
    import pennylane as qml
    from pennylane.templates import StronglyEntanglingLayers
except Exception:
    qml = None
    StronglyEntanglingLayers = None


@dataclass
class QAGConfig:
    num_qubits: int = 8
    q_layers: int = 2
    diff_method: str = "backprop"  # pennylane diff method
    use_lightning: bool = False
    diff_device: str = "cpu"       # just informational
    scale_range: float = 0.5       # final scales = 1.0 + scale_range * tanh(...)
    reduce: str = "mean"           # how to form summary: "mean" or "last"


class NoOpQuantumGate(nn.Module):
    """
    Fallback gate used when qcfg is None or when PennyLane isn't desired.
    Behaves like "no gating": returns ones for summary -> scales, and returns q,k,v unchanged.
    """
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, *args):
        if len(args) == 1:
            summary = args[0]
            B = summary.size(0)
            device = summary.device
            dtype = summary.dtype
            return torch.ones(B, self.n_heads, device=device, dtype=dtype)
        elif len(args) == 3:
            q, k, v = args
            return q, k, v
        else:
            raise ValueError("NoOpQuantumGate expects either (summary) or (q,k,v)")


class QuantumAttentionGate(nn.Module):
    """
    Quantum-like Attention Gate that can be used two ways:
      1) scales = gate(summary_vec)         where summary_vec: [B, H] -> returns [B, n_heads]
      2) q,k,v = gate(q,k,v)                where q,k,v: [B, n_heads, T, head_dim] -> returns gated q,k,v

    NOTE: This implementation uses PennyLane. If PennyLane is not installed and you try to instantiate
    this class, it will raise. Use NoOpQuantumGate when you want no quantum behavior.
    """
    def __init__(self, in_dim: int, n_heads: int, cfg: QAGConfig):
        super().__init__()
        if qml is None:
            raise RuntimeError("PennyLane not available. Install pennylane to use QuantumAttentionGate.")
        self.cfg = cfg
        self.n_heads = n_heads
        self.in_dim = in_dim  # hidden dim H

        # CPU float32 linear layers for angle projection and post-processing
        # keep them on CPU and float32 intentionally
        self.to_angles = nn.Linear(in_dim, cfg.num_qubits, dtype=torch.float32)
        self.post = nn.Linear(cfg.num_qubits, n_heads, dtype=torch.float32)

        # PQC parameters on CPU float32
        q_weights = torch.randn(cfg.q_layers, cfg.num_qubits, 3, dtype=torch.float32) * 0.01
        self.q_weights = nn.Parameter(q_weights, requires_grad=True)

        # Setup Pennylane device (prefer lightning.qubit on CUDA if requested)
        if cfg.use_lightning:
            try:
                # this may raise if lightning not installed or CUDA not available
                self.dev = qml.device("lightning.qubit", wires=cfg.num_qubits, device="cuda")
                print("[QAG] Using pennylane-lightning (cuda) device")
            except Exception as e:
                print(f"[QAG] Warning: unable to use lightning.qubit (cuda): {e}. Falling back to default.qubit (CPU).")
                self.dev = qml.device("default.qubit", wires=cfg.num_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=cfg.num_qubits)

        # Build QNode (returns expectation values for each qubit)
        @qml.qnode(self.dev, diff_method=cfg.diff_method)
        def circuit(angles, weights):
            # angles: length num_qubits
            qml.templates.AngleEmbedding(angles, wires=range(cfg.num_qubits), rotation="Y")
            # StronglyEntanglingLayers expects weights shape (layers, wires, 3)
            StronglyEntanglingLayers(weights, wires=range(cfg.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits)]

        self.circuit = circuit

    def _apply(self, fn):
        """
        Override _apply to keep quantum-related modules on CPU even when the model is moved to CUDA.
        This ensures to_angles, post, and q_weights always remain on CPU for the PQC computation.
        """
        super()._apply(fn)
        # Force these specific modules/parameters back to CPU after any device movement
        self.to_angles = self.to_angles.to("cpu")
        self.post = self.post.to("cpu")
        self.q_weights.data = self.q_weights.data.to("cpu")
        return self

    def _run_pqc_on_batch(self, angles_cpu: torch.Tensor):
        """
        angles_cpu: [B, num_qubits] cpu float32 tensor
        returns: qfeat_cpu [B, num_qubits] cpu float32 tensor (expectation vector per batch)
        """
        outs = []
        # iterate rows (PennyLane QNodes often accept 1D arrays per call)
        for b in range(angles_cpu.size(0)):
            a = angles_cpu[b]              # CPU float32 torch.tensor
            w = self.q_weights.detach()    # Parameter on CPU float32

            try:
                res = self.circuit(a, w)   # may return np.ndarray or torch-like
            except Exception:
                # fallback to numpy inputs (safe)
                res = self.circuit(a.detach().cpu().numpy(), w.detach().cpu().numpy())

            # convert to torch float32 on CPU
            if torch.is_tensor(res):
                ev = res.detach().cpu().float()
            else:
                ev = torch.as_tensor(res, dtype=torch.float32)
            outs.append(ev)

        qfeat_cpu = torch.stack(outs, dim=0)  # [B, num_qubits], float32 on CPU
        return qfeat_cpu

    def forward(self, *args):
        """
        Two behaviors:
          - forward(summary_vec: torch.Tensor[B, H]) -> scales [B, n_heads]
          - forward(q, k, v) where q,k,v are [B, n_heads, T, head_dim] -> gated q,k,v
        """
        if len(args) == 1:
            summary_vec = args[0]
            return self._summary_to_scales(summary_vec)
        elif len(args) == 3:
            q, k, v = args
            return self._gate_qkv(q, k, v)
        else:
            raise ValueError("QuantumAttentionGate.forward expects either (summary) or (q,k,v)")

    def _summary_to_scales(self, summary_vec: torch.Tensor):
        """
        Convert summary [B, H] (maybe on cuda) -> scales [B, n_heads] on same device/dtype as summary_vec
        """
        device_out = summary_vec.device
        dtype_out = summary_vec.dtype

        # Move to CPU float32 (PQC path)
        summary_cpu = summary_vec.detach().to("cpu").float()

        # angles via CPU linear (no grad path here by design: keep linear in module to be trainable though)
        # Keep the forward consistent; allow gradients to pass only through post? This implementation treats PQC as CPU float32 path.
        with torch.no_grad():
            angles_cpu = self.to_angles(summary_cpu)  # [B, num_qubits] float32 CPU

        # Run PQC per batch to get expectation features on CPU
        qfeat_cpu = self._run_pqc_on_batch(angles_cpu)  # [B, num_qubits] float32 CPU

        # Post projection and compute head-wise scales (CPU float32)
        head_logits_cpu = self.post(qfeat_cpu)  # [B, n_heads] CPU float32
        scales_cpu = 1.0 + self.cfg.scale_range * torch.tanh(head_logits_cpu)  # [B, n_heads] CPU float32

        # Move scales to the original device & dtype
        scales = scales_cpu.to(device_out, non_blocking=True).to(dtype=dtype_out)
        return scales  # [B, n_heads] on summary_vec.device/dtype

    def _gate_qkv(self, q, k, v):
        """
        q,k,v: [B, n_heads, T, head_dim]
        Reconstruct summary [B, H] from q (permute+reshape), produce scales, and apply to q,k,v.
        """
        B, n_heads, T, head_dim = q.shape
        # sanity check
        if n_heads != self.n_heads:
            # allow mismatch but warn (we will still proceed)
            # You could raise if you want strictness
            print(f"[QAG] Warning: gate expected n_heads={self.n_heads} but got n_heads={n_heads}")

        # Reconstruct hidden representation [B, T, H]
        H = n_heads * head_dim
        # permute to [B, T, n_heads, head_dim] then reshape last two dims -> H
        q_ = q.permute(0, 2, 1, 3).contiguous().view(B, T, H)  # [B, T, H]

        # Build summary: mean over sequence (or last token)
        if self.cfg.reduce == "mean":
            summary = q_.mean(dim=1)  # [B, H]
        else:
            summary = q_[:, -1, :]    # [B, H]

        # Get scales [B, n_heads] on same device/dtype as summary
        scales = self._summary_to_scales(summary)  # moves to same device/dtype as summary

        # Ensure dtype matches q/k/v dtype (common case: float16 or float32)
        scales = scales.to(dtype=q.dtype)

        # Reshape scales to broadcast: [B, 1, n_heads, 1]
        scales_b = scales.view(B, 1, n_heads, 1)

        # Apply gating to q,k,v
        qg = q * scales_b
        kg = k * scales_b
        vg = v * scales_b

        return qg, kg, vg


# -----------------------------
# GPT-2 style attention classes
# -----------------------------

class GPT2Attention(nn.Module):
    def __init__(self, config, qcfg: Optional[QAGConfig] = None):
        """
        If qcfg is provided, we instantiate a real QuantumAttentionGate (requires PennyLane).
        If qcfg is None, we instantiate NoOpQuantumGate to keep behavior identical to no gating.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5
        self.register_buffer('mask', torch.tril(torch.ones(1, 1, self.seq_len, self.seq_len)))
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # quantum gate (real or no-op)
        if qcfg is None:
            self.qag = NoOpQuantumGate(self.n_heads, self.head_size)
        else:
            self.qag = QuantumAttentionGate(in_dim=self.embed_dim, n_heads=self.n_heads, cfg=qcfg)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        # ---- Quantum gating ----
        # Build a summary from x (pre-attention hidden states), shape [B, H]
        summary = x.mean(dim=1)  # [B, H]
        scales = self.qag(summary)  # [B, n_heads] or identity behavior if NoOp

        # reshape to broadcast across head_dim: [B, n_heads, 1, 1]
        scales = scales.view(b, self.n_heads, 1, 1)
        q = q * scales
        k = k * scales
        v = v * scales

        # ---- Standard attention ----
        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)

        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        return out


class GPT2CrossAttention(nn.Module):
    def __init__(self, config, qcfg: Optional[QAGConfig] = None):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # quantum gate (real or no-op)
        if qcfg is None:
            self.qag = NoOpQuantumGate(self.n_heads, self.head_size)
        else:
            self.qag = QuantumAttentionGate(in_dim=self.embed_dim, n_heads=self.n_heads, cfg=qcfg)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, q, k, v):
        b, t, c = q.shape

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(b, q.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, k.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, v.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)

        # ---- Quantum gating ----
        # Build summary from q's reconstructed hidden (or use other summary)
        # Reconstruct hidden-like vector: [B, T, H]
        q_recon = q.permute(0, 2, 1, 3).contiguous().view(b, q.size(2), self.n_heads * self.head_size)
        if isinstance(self.qag, NoOpQuantumGate):
            # NoOp gate expects summary of shape [B, H] too; either works the same
            summary = q_recon.mean(dim=1)
        else:
            if hasattr(self.qag.cfg, "reduce") and self.qag.cfg.reduce == "last":
                summary = q_recon[:, -1, :]
            else:
                summary = q_recon.mean(dim=1)

        scales = self.qag(summary)  # [B, n_heads]
        scales = scales.view(b, self.n_heads, 1, 1)

        q = q * scales
        k = k * scales
        v = v * scales

        # ---- Cross attention ----
        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)

        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        return out


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim * self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config, qcfg: Optional[QAGConfig] = None):
        """
        qcfg: Optional[QAGConfig]. If provided, gates will be quantum (PennyLane). If None, gates are NoOp.
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config, qcfg)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config, qcfg)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class VisionGPT2Model(nn.Module):
    def __init__(self, config, qcfg):
        super().__init__()

        self.config = config
        self.qcfg = qcfg   # save qcfg

        vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches

        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.seq_len, config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config, qcfg) for _ in range(config.depth)]),  # pass qcfg
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _pos_embed(self,x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self,trainable=False):
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)

        for layer in layers:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')

    def unfreeze_gpt_layers(self,):
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)

        for layer in flatten:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True

    @classmethod
    def from_pretrained(self, config, qcfg):
        model = VisionGPT2Model(config, qcfg)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.','cross_attn.','ln_3','cls_token','pos_embed','patch_embed.','.attn.mask']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]

        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)

        return model

    def forward(self,image,input_ids,labels=None):

        image = self.patch_embed(image)
        image = self._pos_embed(image)

        token_embeddings = self.transformer.wte(input_ids) # batch x seq_len
        pos_embs = torch.arange(0,input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings+positional_embeddings)

        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids, image)

        input_ids = self.transformer.ln_f(input_ids)

        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        lm_logits = self.lm_head(input_ids[:,[-1],:])
        return lm_logits

    def generate(self,image,sequence,max_tokens=50,temperature=1.0,deterministic=False):
        for _ in range(max_tokens):
            out = self(image,sequence)
            out = out[:,-1,:] / temperature
            probs = F.softmax(out,dim=-1)
            if deterministic:
                next_token = torch.argmax(probs,dim=-1,keepdim=True)
            else:
                next_token = torch.multinomial(probs,num_samples=1)
            sequence = torch.cat([sequence,next_token],dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        return sequence.cpu().flatten()
     
    

model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768, # 768
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)
train_config = SimpleNamespace(
    epochs = 20,
    freeze_epochs_gpt = 1,
    freeze_epochs_all = 2,
    lr = 1e-4,
    device = 'cuda',
    model_path = Path('./'),
    batch_size = 32
)



# Load the model and move it to the appropriate device
# Create QAGConfig
qcfg = QAGConfig(
    num_qubits=8,
    q_layers=2,
    diff_method="backprop",
    use_lightning=False,
    diff_device="cpu",
    scale_range=0.5,
    reduce="mean"
)

# Instantiate the model
model = VisionGPT2Model.from_pretrained(model_config, qcfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the saved state dict
state_dict = torch.load('./captioner.pt')
model.load_state_dict(state_dict)

def generate_caption( image, max_tokens=200, temperature=1.0, deterministic=False):
    # model.eval()
    gen_tfms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    image = gen_tfms(image=image)['image']
    image = image.unsqueeze(0).to(device)  # Move the input image tensor to the same device as the model
    sequence = torch.ones(1, 1).long().to(device) * tokenizer.bos_token_id

    caption = model.generate(
        image,
        sequence,
        max_tokens=max_tokens,
        temperature=temperature,
        deterministic=deterministic
    )
    caption = tokenizer.decode(caption.cpu().numpy(), skip_special_tokens=True)  # Move the generated caption back to CPU for decoding

    return caption


import os
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Any

# Updated langchain imports for newer versions
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom Sentence Transformer Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# Initialize RAG components
print("Initializing RAG system...")

# Load and process data
loader = CSVLoader(file_path="./Dataset/reports.csv", encoding="utf-8")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
documents = text_splitter.split_documents(data)

# Use free sentence transformer embeddings
embedding_model = SentenceTransformerEmbeddings()

# Check if chroma db already exists, otherwise create it
try:
    db = Chroma(
        persist_directory="./chroma",
        embedding_function=embedding_model,
    )
    print(f"Loaded existing Chroma database with {db._collection.count()} documents")
except:
    print("Creating new Chroma database...")
    db = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory="./chroma"
    )
    print(f"Created Chroma database with {len(documents)} documents")

# Initialize Groq client for RAG (only if API key is available)
try:
    groq_client = Groq()  # Uses GROQ_API_KEY environment variable
    print("Groq client initialized successfully for RAG")
except Exception as e:
    print(f"Warning: Could not initialize Groq client for RAG: {e}")
    groq_client = None

def rag_query(question: str, context_docs: List[Any], groq_client: Groq) -> str:
    """
    Simple RAG implementation using Groq API with retrieved context
    """
    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt with context
    prompt = f"""Based on the following medical report context, answer the question.

Context:
{context}

Question: {question}

Answer (be concise and medical):"""
    
    # Query Groq API
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
        top_p=1,
        stream=False
    )
    
    return completion.choices[0].message.content

print("RAG system initialized successfully!")


import json

def generate_report_simple(image):
    """
    Simplified report generation using direct Groq API calls
    """
    det = False
    t = np.random.uniform(0.5,1.5)
    caption = generate_caption(image,temperature=t,deterministic=det)
    
    # Initialize Groq client
    client = Groq()  # Uses GROQ_API_KEY environment variable
    
    # Define questions for different report sections
    questions = [
        f"Based on this X-ray finding: '{caption}', what would be the clinical indication? Provide a brief medical indication.",
        f"Based on this X-ray finding: '{caption}', what is the radiological impression? Provide a concise medical impression.",
        f"Based on this X-ray finding: '{caption}', write a complete summary of findings. Provide detailed medical findings."
    ]
    
    res = []
    for question in questions:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Updated to current model
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=500,
                top_p=1,
                stream=False,
                stop=None
            )
            answer = completion.choices[0].message.content
            res.append(answer)
        except Exception as e:
            print(f"Error with Groq API: {e}")
            res.append("Error generating response")
    
    return res

def generate_report_with_streaming(image):
    """
    Report generation with streaming like your example
    """
    det = False
    t = np.random.uniform(0.5,1.5)
    caption = generate_caption(image,temperature=t,deterministic=det)
    
    client = Groq()
    
    questions = [
        f"Based on this X-ray finding: '{caption}', what would be the clinical indication? Provide a brief medical indication.",
        f"Based on this X-ray finding: '{caption}', what is the radiological impression? Provide a concise medical impression.",
        f"Based on this X-ray finding: '{caption}', write a complete summary of findings. Provide detailed medical findings."
    ]
    
    res = []
    for i, question in enumerate(questions):
        print(f"\nGenerating {'Indication' if i==0 else 'Impression' if i==1 else 'Summary'}...")
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=500,
                top_p=1,
                stream=True,
                stop=None
            )
            
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="")
                    response_text += content
            
            res.append(response_text)
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"Error with Groq API: {e}")
            res.append("Error generating response")
    
    return res

def generate_report(image):
    """
    Main report generation function - uses RAG approach with fallback to simple API
    """
    # RAG APPROACH
    try:
        det = False
        t = np.random.uniform(0.5,1.5)
        caption = generate_caption(image,temperature=t,deterministic=det)
        res = []
        questions = [
            f"What is the clinical indication for {caption}?",
            f"What is the radiological impression for {caption}?",
            f"Write complete summary of findings for {caption}"
        ]
        
        for question in questions:
            # Retrieve relevant documents from vector store
            relevant_docs = db.similarity_search(question, k=3)
            # Use RAG query with retrieved context
            response = rag_query(question, relevant_docs, groq_client)
            res.append(response)
        
        return res
    except Exception as e:
        print(f"RAG approach failed: {e}")
        print("Falling back to simple Groq API calls...")
        return generate_report_simple(image)

# Call the generate_caption function
path = "./Dataset/images/CXR105_IM-0037-2001.png"
report = generate_report(path)

import textwrap

print("Indication: ",textwrap.fill(report[0]),"\n\n")
print("Impression: ",textwrap.fill(report[1]),"\n\n")
print("Summary of Findings: ",textwrap.fill(report[2]),"\n\n")