import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers


@dataclass
class QAGConfig:
    num_qubits: int = 6
    q_layers: int = 2
    reduce: str = "mean"
    scale_range: float = 0.5
    diff_method: str = "backprop"
    use_lightning: bool = False


class QuantumAttentionGate(nn.Module):
    """
    Quantum gate that maps a summary vector [B, H] -> per-head scales [B, n_heads].
    The PQC runs on CPU (default.qubit) by default. All linear layers & q_weights are kept in float32.
    After PQC, outputs are moved to the target device (e.g., CUDA) to integrate with the LM.
    """
    def __init__(self, in_dim: int, n_heads: int, cfg: QAGConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads
        self.in_dim = in_dim

        # Keep linear projection and post layers (default on CPU, float32)
        self.to_angles = nn.Linear(in_dim, cfg.num_qubits, dtype=torch.float32)
        self.post = nn.Linear(cfg.num_qubits, n_heads, dtype=torch.float32)

        # PQC weights stored as float32 on CPU
        self.q_weights = nn.Parameter(torch.randn(cfg.q_layers, cfg.num_qubits, 3, dtype=torch.float32) * 0.01)

        # PennyLane device: try lightning.qubit (GPU sim) if requested, otherwise default.qubit on CPU
        if cfg.use_lightning:
            try:
                self.dev = qml.device("lightning.qubit", wires=cfg.num_qubits, device="cuda")
                print("[QAG] Using pennylane-lightning (CUDA)")
            except Exception as e:
                print(f"[QAG] Warning: could not use lightning.qubit (cuda): {e} — falling back to default.qubit (CPU)")
                self.dev = qml.device("default.qubit", wires=cfg.num_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=cfg.num_qubits)

        # QNode: no interface kwarg (modern PennyLane auto-infers interface based on input types)
        @qml.qnode(self.dev, diff_method=cfg.diff_method)
        def circuit(angles, weights):
            # angles expected shape: [num_qubits]
            qml.templates.AngleEmbedding(angles, wires=range(cfg.num_qubits), rotation="Y")
            StronglyEntanglingLayers(weights, wires=range(cfg.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.num_qubits)]

        self.circuit = circuit

    def forward(self, summary_vec: torch.Tensor) -> torch.Tensor:
        """
        summary_vec: [B, H] (may be on CUDA)
        Returns scales: [B, n_heads] on same device as summary_vec
        """
        # target device for outputs (likely cuda)
        device_out = summary_vec.device

        # Convert summary to CPU float32 for PQC path
        summary_cpu = summary_vec.detach().to("cpu").float()  # ensure float32

        # Compute angles on CPU using CPU linear layer (float32)
        with torch.no_grad():
            angles_cpu = self.to_angles(summary_cpu)  # [B, num_qubits], float32 on CPU

        # Run PQC per batch row (CPU). PennyLane may return list, numpy array or torch tensor.
        outs = []
        for b in range(angles_cpu.size(0)):
            a = angles_cpu[b]           # CPU float32 tensor
            w = self.q_weights         # CPU float32 Parameter
            try:
                res = self.circuit(a, w)  # may be torch tensor or numpy/list
            except Exception:
                # fallback: call with numpy
                res = self.circuit(a.detach().cpu().numpy(), w.detach().cpu().numpy())

            # convert to torch tensor float32 on CPU
            if torch.is_tensor(res):
                ev = res.detach().cpu().float()
            else:
                ev = torch.as_tensor(res, dtype=torch.float32)

            outs.append(ev)

        qfeat_cpu = torch.stack(outs, dim=0)  # [B, num_qubits] CPU float32

        # Post projection on CPU (float32)
        head_logits_cpu = self.post(qfeat_cpu)  # [B, n_heads] CPU float32
        scales_cpu = 1.0 + self.cfg.scale_range * torch.tanh(head_logits_cpu)  # [B, n_heads] CPU float32

        # Move final scales to the original device & dtype (float32 -> model uses float32 per safe option)
        scales = scales_cpu.to(device_out, non_blocking=True).to(dtype=summary_vec.dtype)
        return scales


class GPTWithQuantumAdapter(nn.Module):
    def __init__(self, model_name: str, qcfg: QAGConfig, freeze_lm: bool = True, load_in_8bit: bool = False):
        super().__init__()
        self.model_name = model_name
        self.qcfg = qcfg
        self.freeze_lm = freeze_lm
        self.load_in_8bit = load_in_8bit

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SAFEST: run LM in float32 so it matches QAG float32 internals
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float32,   # << SAFE: float32 for no dtype mismatch
            "low_cpu_mem_usage": True,
        }
        # Note: load_in_8bit is incompatible with torch_dtype=float32 in typical usage.
        # We ignore load_in_8bit here to keep safety; set load_in_8bit only if you know how to adjust QAG dtype.
        if load_in_8bit:
            print("[GPT+QAG] Warning: load_in_8bit requested but disabled because we force float32 for safety.")
            # do not set load_in_8bit to avoid dtype/bitwidth incompatibilities

        print(f"[GPT+QAG] Loading model {model_name} with kwargs: {model_kwargs}")
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[GPT+QAG] Model loaded.")

        if freeze_lm:
            print("[GPT+QAG] Freezing LM parameters (only QAG trained).")
            for p in self.lm.parameters():
                p.requires_grad = False

        cfg = self.lm.config
        hidden = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None)
        n_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
        n_heads = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads", None)
        if hidden is None or n_layers is None or n_heads is None:
            raise RuntimeError("Could not infer model dims from config; unsupported model layout.")

        print(f"[GPT+QAG] model dims: hidden={hidden}, layers={n_layers}, heads={n_heads}")

        # Build quantum gates (CPU float32)
        self.qgates = nn.ModuleList([QuantumAttentionGate(in_dim=hidden, n_heads=n_heads, cfg=qcfg) for _ in range(n_layers)])

        # Locate transformer blocks (GPT-2/GPT-Neo style)
        blocks = None
        try:
            blocks = self.lm.transformer.h
        except Exception:
            if hasattr(self.lm, "model") and hasattr(self.lm.model, "h"):
                blocks = self.lm.model.h
            elif hasattr(self.lm, "transformer") and hasattr(self.lm.transformer, "blocks"):
                blocks = self.lm.transformer.blocks
            else:
                raise RuntimeError("Could not find transformer blocks for hooking attention.")

        # Register hooks on attention modules
        hooks = []
        for i, block in enumerate(blocks):
            att_mod = None
            for cand in ("attn", "attention", "self_attn"):
                if hasattr(block, cand):
                    att_mod = getattr(block, cand)
                    break
            if att_mod is None:
                raise RuntimeError(f"Could not find attention module inside block {i}. Block keys: {dir(block)}")

            handle = att_mod.register_forward_hook(self._make_attn_hook(i))
            hooks.append(handle)

        self._hooks = hooks
        print(f"[GPT+QAG] Registered hooks on {len(hooks)} attention modules.")

    def _make_attn_hook(self, layer_idx: int):
        gate = self.qgates[layer_idx]
        cfg = self.qcfg

        def hook(module, inputs, outputs):
            """
            Typical signature:
              inputs[0] = hidden_states [B, T, H]
              outputs[0] = attn_output [B, T, H]
            We compute a summary, call PQC (on CPU) and integrate scales back into attn_output.
            """
            try:
                hidden_states = inputs[0]
                attn_output = outputs[0]
            except Exception:
                return outputs

            # compute summary on same device as hidden_states
            if cfg.reduce == "mean":
                summary = hidden_states.mean(dim=1)  # [B, H]
            else:
                summary = hidden_states[:, -1, :]

            # gate(summary) will internally move to CPU and return scales on summary.device
            scales = gate(summary)  # [B, n_heads] with dtype = summary.dtype (float32)

            B, T, H = attn_output.shape

            # infer number of heads
            n_heads = getattr(module, "num_heads", None)
            if n_heads is None:
                n_heads = getattr(module, "n_head", None)
            if n_heads is None:
                return outputs

            head_dim = H // n_heads
            attn_heads = attn_output.view(B, T, n_heads, head_dim)

            # scales is [B, n_heads] on same device as summary; ensure dtype matches attn_output
            scales = scales.to(dtype=attn_output.dtype)

            scales = scales.view(B, 1, n_heads, 1)  # broadcast dims
            gated = attn_heads * scales
            gated = gated.view(B, T, H)

            if isinstance(outputs, tuple):
                out_list = list(outputs)
                out_list[0] = gated
                return tuple(out_list)
            else:
                return gated

        return hook

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 0):
        # Determine a device for inputs: use device of first model parameter
        try:
            first_param_device = next(self.lm.parameters()).device
        except StopIteration:
            first_param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # Move inputs to the model's first parameter device to avoid cross-device warnings
        enc = {k: v.to(first_param_device) for k, v in enc.items()}

        outputs = self.lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(1e-5, temperature),
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Hello from quantum-enhanced decoder")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_qubits", type=int, default=6)
    parser.add_argument("--q_layers", type=int, default=2)
    parser.add_argument("--use_lightning", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--freeze_lm", action="store_true")
    args = parser.parse_args()

    # keep these on: A100 benefits but we use float32 model
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    qcfg = QAGConfig(num_qubits=args.num_qubits, q_layers=args.q_layers, use_lightning=args.use_lightning)

    print("[main] Building GPT+QAG wrapper...")
    wrapper = GPTWithQuantumAdapter(model_name=args.model, qcfg=qcfg, freeze_lm=args.freeze_lm, load_in_8bit=args.load_in_8bit)
    print("[main] Built wrapper.")

    print("[main] Generating...")
    out = wrapper.generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n=== GENERATED ===\n", out)


if __name__ == "__main__":
    main()
