from dataclasses import dataclass
from typing import Optional, List
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class QAGConfig:
    num_qubits: int = 6
    q_layers: int = 2
    reduce: str = "mean"  # how to summarize hidden states: mean | cls | last
    heads_per_block: Optional[int] = None  # auto-infer for GPT‑2
    scale_range: float = 0.5  # scale in [1-scale, 1+scale]
    diff_method: str = "backprop"  # pennylane diff
    device_name: str = "default.qubit"


class QuantumAttentionGate(nn.Module):
    """Quantum gate that outputs per‑head scales given a block summary vector.

    Steps: summary -> proj -> angles -> PQC -> linear -> per‑head scales in [1-s, 1+s].
    """
    def __init__(self, in_dim: int, n_heads: int, cfg: QAGConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads
        self.to_angles = nn.Linear(in_dim, cfg.num_qubits)
        self.post = nn.Linear(cfg.num_qubits, n_heads)
        self.scale_range = cfg.scale_range

        # PQC weights (L, wires, 3)
        self.q_weights = nn.Parameter(0.01 * torch.randn(cfg.q_layers, cfg.num_qubits, 3))

        # small quantum device (CPU)
        self.dev = qml.device(cfg.device_name, wires=cfg.num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method=cfg.diff_method)
        def circuit(angles, weights):
            qml.templates.AngleEmbedding(angles, wires=range(cfg.num_qubits), rotation="Y")
            StronglyEntanglingLayers(weights, wires=range(cfg.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits)]

        self.circuit = circuit

    def forward(self, summary_vec: torch.Tensor) -> torch.Tensor:
        # summary_vec: [B, H]  (hidden size)
        B, _ = summary_vec.shape
        angles = self.to_angles(summary_vec)  # [B, num_qubits]
        outs = []
        for b in range(B):
            ev = self.circuit(angles[b], self.q_weights)  # [num_qubits]
            # ✅ Convert ev into a proper torch tensor before stacking
            ev = torch.tensor(ev, dtype=angles.dtype, device=angles.device)
            outs.append(ev.unsqueeze(0))

        qfeat = torch.cat(outs, dim=0)  # [B, num_qubits]
        head_logits = self.post(qfeat)  # [B, n_heads]
        # squash to scales in (1 - r, 1 + r)
        scales = 1.0 + self.scale_range * torch.tanh(head_logits)
        return scales  # [B, n_heads]



class GPT2WithQuantumGate(nn.Module):
    def __init__(self, model_name: str = "gpt2", freeze_lm: bool = True, qcfg: QAGConfig = QAGConfig()):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        # discover model dims & heads
        hidden = self.lm.config.n_embd
        n_layers = self.lm.config.n_layer
        n_heads = self.lm.config.n_head

        # one quantum gate per block
        self.qgates = nn.ModuleList([
            QuantumAttentionGate(in_dim=hidden, n_heads=n_heads, cfg=qcfg)
            for _ in range(n_layers)
        ])

        self.qcfg = qcfg

        # register forward hooks to scale attention outputs per block
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        for i, block in enumerate(self.lm.transformer.h):
            handle = block.attn.register_forward_hook(self._make_attn_hook(i))
            self._hooks.append(handle)

    def _make_attn_hook(self, layer_idx: int):
        gate = self.qgates[layer_idx]

        def hook(module, inputs, outputs):
            # GPT‑2 attention forward returns: attn_output, attn_weights, present, ...
            # We want to scale attn_output *per head* using a summary of the *input hidden states*.
            # inputs: (hidden_states, layer_past, attention_mask, ...)
            hidden_states = inputs[0]  # [B, T, H]
            attn_output = outputs[0]   # [B, T, H]

            # get per‑head representation by reshaping attn_output
            B, T, H = attn_output.shape
            n_heads = module.num_heads
            head_dim = H // n_heads
            attn_heads = attn_output.view(B, T, n_heads, head_dim)

            # summary vector from *inputs* (more stable than outputs), e.g., mean over tokens
            if self.qcfg.reduce == "mean":
                summary = hidden_states.mean(dim=1)  # [B, H]
            elif self.qcfg.reduce == "last":
                summary = hidden_states[:, -1, :]
            else:  # cls isn't defined for GPT‑2; fallback to mean
                summary = hidden_states.mean(dim=1)

            # quantum scales: [B, n_heads]
            scales = gate(summary)
            scales = scales.view(B, 1, n_heads, 1)  # broadcast over tokens and head_dim

            gated = attn_heads * scales
            gated = gated.view(B, T, H)

            # replace attn_output in the tuple
            new_outputs = list(outputs)
            new_outputs[0] = gated
            return tuple(new_outputs)

        return hook

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8,
                 top_p: float = 0.95, top_k: int = 0, device: Optional[str] = None) -> str:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.lm.eval()

        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = self.lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(1e-5, temperature),
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)


def demo(model_name: str, prompt: str, max_new_tokens: int):
    model = GPT2WithQuantumGate(model_name=model_name, freeze_lm=True)
    text = model.generate(prompt=prompt, max_new_tokens=max_new_tokens)
    print("\n=== OUTPUT (GPT + Quantum Attention Gate) ===\n", text)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--prompt", type=str, default="Hello from a quantum‑enhanced decoder")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()


demo(model_name=args.model, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
