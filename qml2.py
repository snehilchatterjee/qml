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
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gc
import json

import pandas as pd
df = pd.read_csv("Dataset/reports.csv")
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
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token

train_df, val_df = train_test_split(dataFrame,test_size=0.1)
train_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)
print(len(train_df),len(val_df))

class Dataset:
    def __init__(self, df, tfms):
        self.df = df
        self.tfms = tfms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        sample = self.df.iloc[idx,:]
        image = sample['image']
        caption = sample['caption']
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image,input_ids,labels
    
train_ds = Dataset(train_df,train_tfms)
val_ds = Dataset(val_df,valid_tfms)

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


dl = torch.utils.data.DataLoader(train_ds,shuffle=True,batch_size=2,collate_fn=collate_fn)
_,c,l = next(iter(dl))
print(c[0])
print(l[0])

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
    def from_pretrained(self,config,qcfg):
        model = VisionGPT2Model(config,qcfg)
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
    
    

class Trainer:
    def __init__(self,model_config,train_config, dls):

        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        qcfg = QAGConfig(
            num_qubits=8,
            q_layers=2,
            diff_method="backprop",
            use_lightning=False,
            diff_device="cpu",
            scale_range=0.5,
            reduce="mean"
        )

        self.model = VisionGPT2Model.from_pretrained(model_config, qcfg).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scaler = GradScaler()

        self.train_dl, self.val_dl = dls

        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

#         self.sched = get_linear_schedule_with_warmup(self.optim,num_warmup_steps=0,num_training_steps=total_steps)

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss','train_perplexity','val_loss','val_perplexity']] = None

        self.gen_tfms = A.Compose([
            A.Resize(224,224),
            A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
            ToTensorV2()
        ])


    def save_model(self,):
        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()
        torch.save(self.model,self.train_config.model_path/'captioner.pt')


    def load_best_model(self,):
        sd = torch.load(self.train_config.model_path/'captioner.pt')
        self.model.load_state_dict(sd)


    def train_one_epoch(self,epoch):

        prog = tqdm(self.train_dl,total=len(self.train_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image,input_ids,labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()

                prog.set_description(f'train loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        self.metrics.loc[epoch,['train_loss','train_perplexity']] = (train_loss,train_pxp)


    @torch.no_grad()
    def valid_one_epoch(self,epoch):

        prog = tqdm(self.val_dl,total=len(self.val_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image,input_ids,labels)
                running_loss += loss.item()

                prog.set_description(f'valid loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        self.metrics.loc[epoch,['val_loss','val_perplexity']] = (val_loss,val_pxp)

        return val_pxp


    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()


    def fit(self,):

        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:

            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            prog.set_description('validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }


    @torch.no_grad()
    def generate_caption(self,image,max_tokens=50,temperature=1.0,deterministic=False):

        self.model.eval()

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = self.tokenizer.decode(caption.numpy(),skip_special_tokens=True)

        return caption
    

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



train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=False,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)


trainer = Trainer(model_config,train_config,(train_dl,val_dl))

trainer.fit()