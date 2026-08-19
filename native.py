"""Raon-OpenTTS (F5-TTS-style CFM/DiT) inference model, vendored for ComfyUI.

Model code is the official krafton-ai/Raon-OpenTTS implementation (Apache 2.0,
https://github.com/krafton-ai/Raon-OpenTTS, itself built on SWivid/F5-TTS),
trimmed to the inference path: DiT backbone + CFM sampler + sbhifigan16k mel.
Module names are kept exactly so official checkpoints load with strict=True.

Also hosts the ComfyUI/AIMDO integration helpers (castable module conversion,
dtype tagging, safetensors loading) shared by the loader.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# mel spectrogram (sbhifigan16k: speechbrain HiFi-GAN 16 kHz convention)
# --------------------------------------------------------------------------- #


def crop_waveform_to_hop_aligned_length(waveform, n_fft=1024, hop_length=256):
    total_len = waveform.shape[-1]
    target_frames = (total_len - n_fft) // hop_length
    target_len = target_frames * hop_length + n_fft
    return waveform[..., :target_len]


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def get_sb_hifigan_mel_spectrogram(
    target_sample_rate,
    waveform,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
    n_mel_channels=80,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    norm="slaney",
    mel_scale="slaney",
    compression=True,
):
    import torchaudio

    waveform = crop_waveform_to_hop_aligned_length(waveform, n_fft, hop_length)
    audio_to_mel = torchaudio.transforms.Spectrogram(
        hop_length=hop_length, win_length=win_length, n_fft=n_fft,
        power=power, normalized=normalized, center=False,
    ).to(waveform.device)
    mel_transform = torchaudio.transforms.MelScale(
        sample_rate=target_sample_rate, n_stft=n_fft // 2 + 1, n_mels=n_mel_channels,
        f_min=f_min, f_max=f_max, norm=norm, mel_scale=mel_scale,
    ).to(waveform.device)
    spec = audio_to_mel(waveform)
    mel = mel_transform(spec)
    assert mel.shape[1] == n_mel_channels
    if compression:
        mel = dynamic_range_compression(mel)
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        target_sample_rate=16_000,
        mel_spec_type="sbhifigan16k",
    ):
        super().__init__()
        if mel_spec_type != "sbhifigan16k":
            raise ValueError(f"Raon-OpenTTS checkpoints use sbhifigan16k mel, got '{mel_spec_type}'")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.extractor = get_sb_hifigan_mel_spectrogram
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)
        return self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )


# --------------------------------------------------------------------------- #
# model primitives (upstream f5_tts.model.modules)
# --------------------------------------------------------------------------- #


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# rotary positional embedding (vendored from x_transformers, pair-interleaved form).
# `inv_freq` is a persistent buffer so official checkpoints load with strict=True.


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [b n d] pair-duplicated
        return freqs, 1.0

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)  # [b n d] -> [b 1 n d], broadcast over heads
    t_rot, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (rotate_half(t_rot) * freqs.sin() * scale)
    return torch.cat((t_rot, t_unrotated), dim=-1).type(orig_dtype)


apply_rotary_pos_emb = torch.amp.autocast("cuda", enabled=False)(apply_rotary_pos_emb)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)
        return F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        processor: "AttnProcessor",
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def forward(self, x, c=None, mask=None, rope=None, c_rope=None):
        if c is not None:
            raise NotImplementedError("joint attention is not used by Raon-OpenTTS")
        return self.processor(self, x, mask=mask, rope=rope)


class AttnProcessor:
    def __init__(self, pe_attn_head=None, attn_backend="torch", attn_mask_enabled=True, logit_softcapping=None):
        if attn_backend == "flash_attn":
            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError("attn_backend='flash_attn' but flash_attn is not installed")
        elif attn_backend != "torch":
            raise ValueError(f"unsupported attn_backend: {attn_backend}")
        if logit_softcapping is not None and attn_backend == "flash_attn":
            import warnings
            warnings.warn(
                "logit_softcapping is incompatible with flash_attn; falling back to the torch backend.",
                stacklevel=2,
            )
            attn_backend = "torch"
        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled
        self.logit_softcapping = logit_softcapping

    def __call__(self, attn: Attention, x, mask=None, rope=None) -> torch.Tensor:
        batch_size = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None

            if self.logit_softcapping is not None:
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
                attn_weights = torch.tanh(attn_weights / self.logit_softcapping) * self.logit_softcapping
                if attn_mask is not None:
                    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
                attn_weights = F.softmax(attn_weights, dim=-1)
                x = torch.matmul(attn_weights, value)
            else:
                x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            from flash_attn.bert_padding import pad_input, unpad_input

            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query, key, value, q_cu_seqlens, k_cu_seqlens,
                    q_max_seqlen_in_batch, k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=True,
        logit_softcapping=None,
        post_norm=False,
        norm_type="layernorm",
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
                logit_softcapping=logit_softcapping,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

        if post_norm:
            if norm_type == "rmsnorm":
                self.attn_post_norm = RMSNorm(dim, eps=1e-6)
                self.ff_post_norm = RMSNorm(dim, eps=1e-6)
            else:
                self.attn_post_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                self.ff_post_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.attn_post_norm = None
            self.ff_post_norm = None

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        if self.attn_post_norm is not None:
            attn_output = self.attn_post_norm(attn_output)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        if self.ff_post_norm is not None:
            ff_output = self.ff_post_norm(ff_output)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


# --------------------------------------------------------------------------- #
# DiT backbone (upstream f5_tts.model.backbones.dit)
# --------------------------------------------------------------------------- #


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # 0 = filler token
        self.mask_padding = mask_padding
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):
        text = text + 1  # 0 is the filler token (batch pads are -1, see list_str_to_idx)
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long, device=text.device)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed.to(text.dtype)  # freqs_cis stays fp32; cast at use
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)
        return text


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        logit_softcapping=None,
        post_norm=False,
        norm_type="layernorm",
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    logit_softcapping=logit_softcapping,
                    post_norm=post_norm,
                    norm_type=norm_type,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def get_input_embed(self, x, cond, text, drop_audio_cond=False, drop_text=False, cache=True):
        seq_len = x.shape[1]
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x,
        cond,
        text,
        time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cfg_infer=False,
        cache=False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        if cfg_infer:
            x_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        return self.proj_out(x)


# --------------------------------------------------------------------------- #
# CFM sampler (upstream f5_tts.model.cfm, inference path only)
# --------------------------------------------------------------------------- #


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def lens_to_mask(t, length=None):
    if not exists(length):
        length = t.amax()
    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    from torch.nn.utils.rnn import pad_sequence

    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]
    return pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask=(0.7, 1.0),
        vocab_char_map: dict | None = None,
    ):
        super().__init__()
        self.frac_lengths_mask = frac_lengths_mask
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.transformer = transformer
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond,
        text,
        duration,
        *,
        lens=None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed=None,
        max_duration=4096,
        vocoder: Callable | None = None,
        use_epss=True,
        no_ref_audio=False,
        edit_mask=None,
        step_callback: Callable[[int], None] | None = None,
    ):
        self.eval()
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                raise ValueError("CFM.vocab_char_map is not set")
            assert text.shape[0] == batch

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        step_state = {"count": 0}

        def fn(t, x):
            if step_callback is not None:
                step_callback(step_state["count"])
                step_state["count"] += 1
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    drop_audio_cond=False, drop_text=False, cache=True,
                )
                return pred
            pred_cfg = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, cfg_infer=True, cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        from torch.nn.utils.rnn import pad_sequence

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        # The schedule is computed in fp32 and cast once: in bf16 the sway cosine term
        # rounds adjacent timesteps to the same value (cos(x) ~= 1 collapses), which
        # torchdiffeq rejects ("t must be strictly increasing").
        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float32)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float32)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        t = t.to(step_cond.dtype)

        from torchdiffeq import odeint

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        return out, trajectory

    def forward(self, *args, **kwargs):
        raise NotImplementedError("training forward is not part of the ComfyUI build")


# --------------------------------------------------------------------------- #
# text front-end helpers (upstream f5_tts infer utilities)
# --------------------------------------------------------------------------- #


def convert_char_to_pinyin(text_list, polyphone=True):
    import jieba
    from pypinyin import Style, lazy_pinyin

    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

    def is_chinese(c):
        return "㌀" <= c <= "鿿"

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.append(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


def chunk_text(text, max_chars=135):
    import re

    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += (
                sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (
                sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
            )
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def fix_ref_text_ending(ref_text: str) -> str:
    ref_text = (ref_text or "").strip()
    if not ref_text:
        return ref_text
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text


def normalize_peak(audio, eps=1e-3, max_gain=None):
    if isinstance(audio, torch.Tensor):
        max_abs = audio.abs().max()
        max_abs = torch.clamp(max_abs, min=eps)
        gain = 1.0 / max_abs
        if max_gain is not None:
            gain = torch.clamp(gain, max=max_gain)
        audio = audio * gain
        return torch.clamp(audio, -1.0, 1.0)
    raise TypeError(type(audio))


def remove_silence_edges(audio, silence_threshold=-42):
    from pydub import silence

    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    return audio[: int(non_silent_end_duration * 1000)]


def _estimate_ref_seconds_trimmed_segment(
    aseg,
    base_silence_threshold: int = -42,
    target_dbfs: float = -20.0,
    max_gain_db: float = 30.0,
    thr_margin_db: float = 18.0,
) -> float:
    """VAD-trimmed reference length for generation-length estimation (upstream recipe)."""
    from pydub import AudioSegment

    if aseg.dBFS != float("-inf"):
        gain = target_dbfs - aseg.dBFS
        gain = max(min(gain, max_gain_db), -max_gain_db)
        aseg_norm = aseg.apply_gain(gain)
    else:
        aseg_norm = aseg
    if aseg_norm.dBFS != float("-inf"):
        dyn_thr = aseg_norm.dBFS - thr_margin_db
        silence_threshold = min(base_silence_threshold, int(dyn_thr))
    else:
        silence_threshold = base_silence_threshold
    trimmed = remove_silence_edges(aseg_norm, silence_threshold=silence_threshold)
    trimmed = trimmed + AudioSegment.silent(duration=50)
    if len(trimmed) < 80:
        return aseg.duration_seconds
    return trimmed.duration_seconds


def estimate_ref_seconds_trimmed(ref_audio_path: str, **kwargs) -> float:
    from pydub import AudioSegment

    return _estimate_ref_seconds_trimmed_segment(AudioSegment.from_file(ref_audio_path), **kwargs)


def estimate_ref_seconds_trimmed_tensor(waveform: torch.Tensor, sample_rate: int, **kwargs) -> float:
    """Same VAD estimate from an in-memory [1, n] or [n] waveform (no temp files)."""
    from pydub import AudioSegment

    wav = waveform.detach().float().cpu()
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    pcm16 = (wav.clamp(-1.0, 1.0) * 32767.0).short().numpy().tobytes()
    aseg = AudioSegment(data=pcm16, sample_width=2, frame_rate=int(sample_rate), channels=1)
    return _estimate_ref_seconds_trimmed_segment(aseg, **kwargs)


# --------------------------------------------------------------------------- #
# ComfyUI audio helpers
# --------------------------------------------------------------------------- #


def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform[0].detach().float().cpu()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    elif wav.ndim == 2:
        wav = wav.squeeze(0)
    return wav.contiguous(), sample_rate


def tensor_audio_to_comfy(audio: torch.Tensor, sample_rate: int) -> dict:
    audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
    return {"waveform": audio.view(1, 1, -1).contiguous(), "sample_rate": int(sample_rate)}


def cross_fade(seg_a: torch.Tensor, seg_b: torch.Tensor, fade_len: int) -> torch.Tensor:
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)
    fade_len = int(min(fade_len, seg_a.shape[1], seg_b.shape[1]))
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)
    ramp = torch.linspace(0.0, 1.0, fade_len, device=seg_a.device, dtype=seg_a.dtype).view(1, -1)
    overlap = seg_a[:, -fade_len:] * (1.0 - ramp) + seg_b[:, :fade_len] * ramp
    return torch.cat([seg_a[:, :-fade_len], overlap, seg_b[:, fade_len:]], dim=1)


@contextlib.contextmanager
def attention_runtime(attention: str):
    """Temporarily route F.scaled_dot_product_attention through sageattention."""
    if attention != "sageattention":
        yield
        return
    from sageattention import sageattn

    original_sdpa = F.scaled_dot_product_attention

    def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (attn_mask is not None or dropout_p not in (0, 0.0) or query.device.type != "cuda"
                or query.dtype not in (torch.float16, torch.bfloat16)):
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, **kwargs)
        try:
            output = sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            return output[0] if isinstance(output, tuple) else output
        except Exception:
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, **kwargs)

    F.scaled_dot_product_attention = sage_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


# --------------------------------------------------------------------------- #
# ComfyUI/AIMDO castable module conversion
# --------------------------------------------------------------------------- #


class _ComfyLinear(nn.Linear):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, x):
        from comfy.ops import cast_bias_weight, uncast_bias_weight

        if not hasattr(self, "_v") and self.weight.device == x.device:
            return F.linear(x, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, x, offloadable=True)
        try:
            return F.linear(x, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyEmbedding(nn.Embedding):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, input):
        from comfy.ops import cast_bias_weight, uncast_bias_weight

        if not hasattr(self, "_v") and self.weight.device == input.device:
            return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        weight, bias, stream = cast_bias_weight(self, dtype=self._weight_dtype(), device=input.device, offloadable=True)
        try:
            return F.embedding(input, weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConv1d(nn.Conv1d):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input):
        from comfy.ops import cast_bias_weight, uncast_bias_weight

        if (not hasattr(self, "_v") and self.weight.device == input.device
                and (self.bias is None or self.bias.device == input.device)):
            return self._conv_forward(input, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return self._conv_forward(input, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Patch castable modules in-place so DynamicVRAM can page their weights."""
    for module in model.modules():
        if isinstance(module, (_ComfyLinear, _ComfyEmbedding, _ComfyConv1d)):
            continue
        if isinstance(module, nn.Linear):
            module.__class__ = _ComfyLinear
        elif type(module) is nn.Embedding:
            module.__class__ = _ComfyEmbedding
        elif isinstance(module, nn.Conv1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConv1d


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag floating tensors with the dtype Comfy/AIMDO should materialize.

    INT8 ConvRot weights are never tagged (not floating), and per-row weight
    scales stay fp32 so the quantized kernels receive exact scales.
    """
    for sub in module.modules():
        for name, value in sub.named_parameters(recurse=False):
            if value is not None and value.is_floating_point() and not name.endswith(("weight_scale", "inv_freq", "freqs_cis")):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)
        for name, value in sub.named_buffers(recurse=False):
            if value is not None and value.is_floating_point() and not name.endswith(("weight_scale", "inv_freq", "freqs_cis")):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)


# --------------------------------------------------------------------------- #
# weight loading
# --------------------------------------------------------------------------- #


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, dtype: torch.dtype | None) -> None:
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device

        # dtype=tensor.dtype: without it accelerate casts the value back to the
        # (fp32) meta-initialized parameter dtype. Integer tensors are never cast.
        set_module_tensor_to_device(module, name, device="cpu", value=tensor.contiguous(), dtype=tensor.dtype)
        return
    except ImportError:
        pass
    target = dict(module.named_parameters(remove_duplicate=False)).get(name)
    if target is None:
        target = dict(module.named_buffers(remove_duplicate=False)).get(name)
    if target is None:
        raise KeyError(name)
    if target.shape != tensor.shape:
        raise ValueError(f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    target.data = tensor.contiguous()


def load_safetensors_file(model: nn.Module, path: Path, dtype_policy=None) -> None:
    """Load a single safetensors file into model, casting floats per dtype_policy(name).

    Keys ending in .comfy_quant are quantization metadata consumed by the int8
    runtime (see int8.py), never module tensors; they are skipped here.
    """
    from safetensors import safe_open

    param_names = set(dict(model.named_parameters(remove_duplicate=False)))
    buffer_names = set(dict(model.named_buffers(remove_duplicate=False)))
    loaded: set[str] = set()
    unexpected: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.endswith(".comfy_quant"):
                continue
            if name not in param_names and name not in buffer_names:
                unexpected.append(name)
                continue
            target_dtype = dtype_policy(name) if dtype_policy is not None else None
            _set_tensor(model, name, f.get_tensor(name), target_dtype)
            loaded.add(name)
    missing = [name for name in (param_names | buffer_names) - loaded
               if not name.endswith(("freqs_cis", "dummy"))]
    if missing:
        raise RuntimeError(f"Weights missing from {path.name}: {len(missing)} tensor(s), first: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected tensors in {path.name}: {unexpected[:8]}")
    _materialize_buffers(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _materialize_buffers(model: nn.Module) -> None:
    """Recompute deterministic non-persistent buffers absent from the checkpoint."""
    for module in model.modules():
        if isinstance(module, TextEmbedding) and module.extra_modeling:
            module.freqs_cis = precompute_freqs_cis(module.text_embed.embedding_dim, module.precompute_max_pos)
        elif isinstance(module, MelSpec):
            module.dummy = torch.tensor(0)
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is not None and buf.is_meta:
            raise RuntimeError(f"Buffer {name} is still on the meta device after weight loading.")
