#!/usr/bin/env python3
"""llmfit_lib.py - introspection, ceiling math and log parsing for llmfit.sh.

Subcommands
  probe   read device + GGUF facts, determine memory bandwidth, print the
          theoretical decode/prefill ceilings and a speculative-decoding
          prediction table
  parse   turn one llama-server log into a summary.tsv row
  report  aggregate a sweep directory into a ranked report plus a
          ready-to-run llama-server command

stdlib only. No dependency on anything outside this directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------

# type id -> (elements per block, bytes per block).  Sizes must sum to the
# file size within a few percent, which probe verifies and warns about.
GGML_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 4: (32, 20), 5: (32, 26),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84),
    11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
    28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17),
}
GGML_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}


def component_of(name: str) -> str:
    """Coarse component a tensor belongs to.

    Ordered most-specific first: a tensor named `attn_norm` is a norm, and
    `ffn_gate_exps` is expert weight, not dense FFN.
    """
    if name == "token_embd.weight" or name.endswith("token_embd.weight"):
        return "embeddings"
    if name.startswith("output"):
        return "output head"
    if "nextn" in name:
        return "mtp head"
    if "_exps" in name or "_expert" in name or name.endswith("_shexp.weight"):
        return "experts"
    if "ffn_gate_inp" in name or "ffn_gate_tid" in name:
        return "router"
    if "ssm_" in name:
        return "ssm (recurrent)"
    if name.endswith("_norm.weight") or name.endswith("norm.weight"):
        return "norms"
    if ".attn_" in name or name.startswith("attn_") or "inp_gate" in name:
        return "attention"
    if ".ffn_" in name or name.startswith("ffn_"):
        return "ffn (dense)"
    if "per_layer" in name:
        return "per-layer"
    return "other"


def _layer_signature(names: list) -> tuple:
    """The set of tensor roles in a block, used to group identical layers."""
    return tuple(sorted(n.split(".", 2)[2] for n in names if n.count(".") >= 2))


def kv_layer_count(arch: str, kv: dict, n_layer: int,
                   exclude: int | None = None) -> int:
    """Number of layers that hold a growing KV cache in the main context.

    Two corrections over "every layer holds KV":
      * hybrid stacks mix recurrent (SSM / linear-attention) layers, which keep
        a fixed-size state, with full-attention layers that keep KV.  Counting
        all layers overstated KV by the attention:recurrent ratio (65/16 = 4x
        on the test model).
      * the MTP (nextn) block is driven by its own draft context, so it does
        not take a slice of the target KV cache.  Verified against llama.cpp,
        which reports "16 layers" for a stack whose metadata marks 17
        attention layers.
    """
    rec = kv.get(f"{arch}.attention.recurrent_layers")
    if isinstance(rec, list) and rec:
        return max(1, sum(1 for i, x in enumerate(rec) if not x and i != exclude))
    shared = kv.get(f"{arch}.attention.shared_kv_layers")
    if isinstance(shared, int) and shared > 0:
        return max(1, n_layer - shared)
    return n_layer or 1


def layer_pattern(arch: str, kv: dict, n_layer: int,
                  mtp: int | None = None) -> str:
    """Compact A/R string: A holds KV, R is recurrent, M is the MTP block."""
    rec = kv.get(f"{arch}.attention.recurrent_layers")
    if isinstance(rec, list) and rec:
        return "".join(
            "M" if i == mtp else ("R" if x else "A")
            for i, x in enumerate(rec))
    shared = kv.get(f"{arch}.attention.shared_kv_layers")
    if isinstance(shared, int) and shared > 0:
        return "A" * max(1, n_layer - shared) + "r" * shared
    return "A" * (n_layer or 0)


def _gguf_reader(f):
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        n = u64()
        return f.read(n).decode("utf-8", "replace")

    def value(t):
        if t == 8:
            return string()
        if t == 9:  # array
            et = u32()
            n = u64()
            return [value(et) for _ in range(n)]
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    return u32, u64, string, value


def read_gguf(path: str) -> dict:
    """Return {'kv': {...}, 'tensors': [(name, type, dims, offset)], 'version': n}."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        u32, u64, string, value = _gguf_reader(f)
        version = u32()
        n_tensors = u64()
        n_kv = u64()
        kv = {}
        for _ in range(n_kv):
            k = string()
            kv[k] = value(u32())
        tensors = []
        for _ in range(n_tensors):
            name = string()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            ttype = u32()
            offset = u64()
            tensors.append((name, ttype, dims, offset))
    return {"kv": kv, "tensors": tensors, "version": version}


def tensor_bytes(ttype: int, dims: list) -> int | None:
    if ttype not in GGML_TYPES:
        return None
    blk, tsz = GGML_TYPES[ttype]
    n = 1
    for d in dims:
        n *= d
    return (n // blk) * tsz


def model_facts(path: str) -> dict:
    """Architecture, size and per-token traffic facts for a GGUF model."""
    g = read_gguf(path)
    kv, tensors = g["kv"], g["tensors"]

    def k(name, default=None):
        return kv.get(name, default)

    arch = k("general.architecture", "unknown")

    def ak(suffix, default=None):
        return kv.get(f"{arch}.{suffix}", default)

    n_layer = ak("block_count", 0)
    n_embd = ak("embedding_length", 0)
    n_ff = ak("feed_forward_length", 0)
    n_head = ak("attention.head_count", 0)
    n_head_kv = ak("attention.head_count_kv", n_head)
    k_len = ak("attention.key_length", n_embd // n_head if n_head else 0)
    v_len = ak("attention.value_length", k_len)
    n_ctx_train = ak("context_length", 0)
    n_expert = ak("expert_count", 0)
    n_expert_used = ak("expert_used_count", n_expert)
    n_nextn = ak("nextn_predict_layers", 0)

    total_b = 0
    unknown_types = set()
    n_vocab = 0
    out_head_b = 0
    out_head_type = None
    token_embd_b = 0
    expert_b = 0
    nonexpert_b = 0
    mtp_b = 0
    mtp_idx = None

    for name, ttype, dims, _off in tensors:
        b = tensor_bytes(ttype, dims)
        if b is None:
            unknown_types.add(ttype)
            continue
        total_b += b
        if "expert" in name:
            expert_b += b
        else:
            nonexpert_b += b
        if name in ("output.weight", "output.bias") and len(dims) == 2:
            n_vocab = max(n_vocab, dims[1])
            out_head_b = b
            out_head_type = ttype
        if name == "token_embd.weight" and len(dims) == 2:
            n_vocab = max(n_vocab, dims[1])
            token_embd_b = b
        if ".nextn." in name:
            mtp_idx = int(name.split(".")[1]) if name.startswith("blk.") else None

    # Many models tie the LM head to the embedding matrix (no `output.weight`).
    # The head is still read once per drafted token during speculation, so a
    # zero here would badly understate draft cost.
    tied_head = False
    if out_head_b == 0 and token_embd_b:
        out_head_b = token_embd_b
        tied_head = True

    # The MTP head lives in its own trailing block; count that whole block.
    if mtp_idx is not None:
        pref = f"blk.{mtp_idx}."
        mtp_b = 0
        for name, ttype, dims, _ in tensors:
            if name.startswith(pref):
                b = tensor_bytes(ttype, dims)
                if b:
                    mtp_b += b

    n_params = sum(
        (lambda d: __import__("math").prod(d))(dims) for _n, _t, dims, _o in tensors
    )
    n_params_declared = k("general.parameter_count")

    # ---- structural breakdown -------------------------------------------
    type_hist, comp_b, block_names, block_b = {}, {}, {}, {}
    for name, ttype, dims, _ in tensors:
        b = tensor_bytes(ttype, dims)
        if b is None:
            continue
        tn = GGML_NAMES.get(ttype, str(ttype))
        type_hist[tn] = type_hist.get(tn, 0) + 1
        comp = component_of(name)
        comp_b[comp] = comp_b.get(comp, 0) + b
        if name.startswith("blk."):
            idx = int(name.split(".")[1])
            block_names.setdefault(idx, []).append(name)
            block_b[idx] = block_b.get(idx, 0) + b
    # group layers that share an identical set of tensor roles: this is what
    # makes a hybrid stack (attention vs recurrent) visible at a glance
    sig_to_idx = {}
    for idx, names in block_names.items():
        sig_to_idx.setdefault(_layer_signature(names), []).append(idx)
    archetypes = []
    for sig, idxs in sorted(sig_to_idx.items(), key=lambda kv_: -len(kv_[1])):
        archetypes.append({
            "count": len(idxs),
            "indices": sorted(idxs),
            "roles": list(sig),
            "bytes": sum(block_b.get(i, 0) for i in idxs),
        })
    n_recurrent = sum(
        1 for i in block_names
        if any("ssm_" in n for n in block_names[i]))
    n_attn = sum(
        1 for i in block_names
        if i != mtp_idx
        and any((".attn_q" in n or ".attn_qkv" in n) for n in block_names[i])
        and not any("ssm_" in n for n in block_names[i]))

    # Which layers hold KV, and of those, which are full-attention vs windowed?
    # A layer's attn_k output width answers both: absent attn_k means the layer
    # keeps no KV of its own (recurrent, or sharing another layer's), and the
    # width is n_head_kv * key_length for full attention or
    # n_head_kv * key_length_swa for a windowed layer. Verified against
    # llama.cpp on gemma-4-E4B, which reports a 4-layer non-SWA cache and a
    # 20-layer SWA cache -- exactly what the tensor shapes say.
    kv_blocks = {}
    for name, ttype, dims, _ in tensors:
        if (name.startswith("blk.") and len(dims) == 2
                and (name.endswith(".attn_k.weight")
                     or name.endswith(".attn_v.weight"))):
            kv_blocks[int(name.split(".")[1])] = dims[1]
    kv_blocks.pop(mtp_idx, None)
    k_swa = kv.get(f"{arch}.attention.key_length_swa", k_len)
    v_swa = kv.get(f"{arch}.attention.value_length_swa", v_len)
    if kv_blocks:
        n_kv_layers = len(kv_blocks)
        dim_full = n_head_kv * k_len
        dim_swa = n_head_kv * k_swa
        n_full_attn = sum(1 for d in kv_blocks.values() if d == dim_full)
        n_swa_attn = sum(1 for d in kv_blocks.values() if d == dim_swa)
        if n_full_attn + n_swa_attn != n_kv_layers:
            n_full_attn, n_swa_attn = n_kv_layers, 0   # shapes inconclusive
    else:
        n_kv_layers = kv_layer_count(arch, kv, n_layer, exclude=mtp_idx)
        n_full_attn, n_swa_attn = n_kv_layers, 0

    if n_expert and n_expert > 0 and expert_b:
        frac = (n_expert_used or n_expert) / n_expert
        active_b = nonexpert_b + int(expert_b * frac)
    else:
        active_b = total_b

    return {
        "path": path,
        "arch": arch,
        "name": k("general.name", ""),
        "size_label": k("general.size_label", ""),
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_ff": n_ff,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "k_len": k_len,
        "v_len": v_len,
        "n_ctx_train": n_ctx_train,
        "n_vocab": n_vocab,
        "n_expert": n_expert or 0,
        "n_expert_used": n_expert_used or 0,
        "n_nextn": n_nextn or 0,
        "has_mtp": bool(mtp_idx is not None),
        "mtp_block": mtp_idx,
        "n_params": n_params,
        "n_params_declared": n_params_declared,
        "file_bytes": os.path.getsize(path),
        "tensor_bytes": total_b,
        "active_bytes": active_b,
        "expert_bytes": expert_b,
        "nonexpert_bytes": nonexpert_b,
        "output_head_bytes": out_head_b,
        "output_head_type": ("tied to token_embd" if tied_head
                             else GGML_NAMES.get(out_head_type, str(out_head_type))),
        "tied_head": tied_head,
        "mtp_bytes": mtp_b,
        "unknown_types": sorted(unknown_types),
        "type_hist": dict(sorted(type_hist.items(), key=lambda kv_: -kv_[1])),
        "component_bytes": dict(sorted(comp_b.items(), key=lambda kv_: -kv_[1])),
        "archetypes": archetypes,
        "n_blocks": len(block_names),
        "n_recurrent_layers": n_recurrent,
        "n_attn_layers": n_attn,
        # Derived from attention tensor shapes where possible: it is exact and
        # immune to the metadata quirks (shared KV, hybrid recurrent stacks,
        # interleaved sliding-window attention) that make heuristics wrong.
        "n_kv_layers": n_kv_layers,
        "n_full_attn_layers": n_full_attn,
        "n_swa_attn_layers": n_swa_attn,
        "k_len_swa": k_swa,
        "v_len_swa": v_swa,
        "swa_window": kv.get(f"{arch}.attention.sliding_window") or 0,
        "layer_pattern": layer_pattern(arch, kv, n_layer, mtp=mtp_idx),
        "kv": {kk: vv for kk, vv in kv.items()
               if isinstance(vv, (int, float, str)) and len(str(vv)) < 120},
    }


# --------------------------------------------------------------------------
# devices + memory bandwidth
# --------------------------------------------------------------------------

def run(cmd: list, **kw) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def enumerate_devices(*binaries: str) -> dict:
    """Devices the build supports, plus the Vulkan `uma` flag.

    Takes several binaries because they do not report the same detail:
    llama-bench prints the `uma:` flag and the loaded-backend list, while
    llama-server only prints the device table. Merge whatever each provides.
    """
    out = ""
    for b in binaries:
        if not b:
            continue
        o = run([b, "--list-devices"])
        if "Available devices:" in o:
            out += o
    by_id = {}
    for m in re.finditer(r"^\s+([A-Za-z]+\d+):\s+(.+?)\s+\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$",
                         out, re.M):
        did = m.group(1)
        entry = {
            "id": did,
            "name": m.group(2),
            "total_mib": int(m.group(3)),
            "free_mib": int(m.group(4)),
            "backend": re.match(r"[A-Za-z]+", did).group(0),
        }
        # merge duplicates: keep the largest reported total
        if did not in by_id or entry["total_mib"] > by_id[did]["total_mib"]:
            by_id[did] = entry
    devices = list(by_id.values())
    uma = None
    for m in re.finditer(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*\|\s*uma:\s*(\d)", out):
        uma = int(m.group(2))
    backends = sorted(set(re.findall(r"load_backend: loaded (\w+) backend", out)))
    return {"devices": devices, "uma": uma, "backends": backends}


def _dmidecode_dimms() -> list:
    out = run(["sudo", "-n", "dmidecode", "-t", "memory"])
    if "Memory Device" not in out:
        out = run(["dmidecode", "-t", "memory"])
    dimms = []
    for blk in out.split("Memory Device")[1:]:
        size = re.search(r"^\s*Size:\s*(\d+)\s*(GiB|GB|MiB|MB)", blk, re.M)
        if not size:
            continue
        speed = re.search(r"^\s*Configured Memory Speed:\s*(\d+)\s*MT/s", blk, re.M) \
            or re.search(r"^\s*Speed:\s*(\d+)\s*MT/s", blk, re.M)
        mtype = re.search(r"^\s*Type:\s*(\S+)", blk, re.M)
        dimms.append({
            "size": size.group(1) + " " + size.group(2),
            "mt_s": int(speed.group(1)) if speed else None,
            "type": mtype.group(1) if mtype else "?",
        })
    return dimms


def detect_bandwidth(dev: dict, overrides: dict) -> dict:
    """Best-effort theoretical peak bandwidth, plus the sources used.

    Only the DDR path (integrated / UMA) is derived from hardware data.  A
    discrete board's memory bus width is not exposed by sysfs, so it needs
    --mem-bus-bits (and optionally --mem-mt-s); otherwise the theoretical
    figure is reported as unknown and the harness relies on the measured
    effective bandwidth from the raw benchmark.
    """
    res = {"theoretical_gbs": None, "source": "unknown", "detail": [], "dimm": []}

    if overrides.get("bus_bits") and overrides.get("mt_s"):
        bits, mt = overrides["bus_bits"], overrides["mt_s"]
        res["theoretical_gbs"] = bits / 8 * mt * 1e6 / 1e9
        res["source"] = "override"
        res["detail"].append(f"override: {bits}-bit @ {mt} MT/s")
        return res

    # Integrated: memory clock is the system RAM clock, and each DDR5/DDR4
    # DIMM contributes a 64-bit data path.
    if dev.get("uma") == 1:
        dimms = _dmidecode_dimms()
        res["dimm"] = dimms
        speeds = [d["mt_s"] for d in dimms if d["mt_s"]]
        if dimms and speeds:
            mt = max(speeds)
            bits = 64 * len(dimms)
            res["theoretical_gbs"] = bits / 8 * mt * 1e6 / 1e9
            res["source"] = "dmidecode (UMA: DIMMs x 64-bit)"
            res["detail"].append(
                f"{len(dimms)} x {dimms[0]['type']} @ {mt} MT/s -> {bits}-bit")
            return res
        # Fall back to the GPU's own memory clock: DDR transfers twice per clock.
        mclk = _max_clk("/sys/class/drm/card*/device/pp_dpm_mclk")
        if mclk:
            res["detail"].append(f"GPU mclk {mclk} MHz (system RAM clock)")
            res["source"] = "pp_dpm_mclk only - bus width unknown"
        return res

    # Discrete AMD
    mclk = _max_clk("/sys/class/drm/card*/device/pp_dpm_mclk")
    if mclk:
        res["detail"].append(f"GPU mclk {mclk} MHz")
    smi = run(["rocm-smi", "--showclocks"])
    m = re.search(r"mclk clock level: \d+: \((\d+)Mhz\)", smi)
    if m:
        res["detail"].append(f"rocm-smi mclk {m.group(1)} MHz")
    if mclk and overrides.get("bus_bits"):
        bits = overrides["bus_bits"]
        res["theoretical_gbs"] = bits / 8 * (2 * mclk) * 1e6 / 1e9
        res["source"] = "pp_dpm_mclk x --mem-bus-bits"
        res["detail"].append(f"{bits}-bit @ {2 * mclk} MT/s (DDR)")
    else:
        res["source"] = "discrete: bus width not in sysfs - pass --mem-bus-bits"
    return res


def _max_clk(pattern: str) -> int | None:
    best = None
    for p in glob.glob(pattern):
        try:
            txt = open(p).read()
        except OSError:
            continue
        for m in re.finditer(r":\s*(\d+)Mhz", txt):
            v = int(m.group(1))
            best = v if best is None else max(best, v)
    return best


# --------------------------------------------------------------------------
# ceilings + speculative prediction
# --------------------------------------------------------------------------

def kv_profile(f: dict, cache_type: str, ctx: int, parallel: int = 1) -> dict:
    """KV accounting, split by attention type.

    Three effects are easy to miss and each misstates KV severalfold:

    * hybrid stacks have recurrent layers holding no KV at all;
    * interleaved sliding-window attention (iSWA) splits the rest into
      full-attention layers that grow with the context and windowed layers
      capped by the sliding window;
    * the two groups can use different head dimensions (gemma-4: 512 for full,
      256 for windowed, via key_length_swa).

    Verified against llama.cpp for gemma-4-E4B: 4 full layers x 32768 cells and
    20 windowed layers x 2560 cells come to 172 MiB, which this reproduces,
    where treating all 24 KV layers as full-context overstates it 5x.
    """
    blk, tsz = GGML_TYPES.get(_cache_type_id(cache_type), (1, 2))
    bpe = tsz / blk
    arch, kv = f.get("arch", ""), f.get("kv", {})
    hkv = f.get("n_head_kv") or 1
    k_full, v_full = f.get("k_len") or 0, f.get("v_len") or 0
    k_swa = f.get("k_len_swa", k_full)
    v_swa = f.get("v_len_swa", v_full)
    swa = f.get("swa_window", 0)
    n_kv = f.get("n_kv_layers") or f.get("n_layer") or 1
    n_full = f.get("n_full_attn_layers", n_kv)
    n_swa = f.get("n_swa_attn_layers", max(0, n_kv - n_full))

    per_full = hkv * (k_full + v_full) * bpe
    per_swa = hkv * (k_swa + v_swa) * bpe
    window = min(ctx, swa) if swa else ctx
    # a generated token reads the whole prefix in full-attention layers and
    # only the window in windowed ones
    read = n_full * per_full * ctx + n_swa * per_swa * window
    # Allocation is not the same as traffic: llama.cpp sizes the windowed cache
    # at window x (parallel + 1) cells.  Verified against its reported buffers:
    # 512x2 = 1024 cells (11.25 MiB) at --parallel 1 and 512x5 = 2560 cells
    # (28.12 MiB) at 4 slots.
    cells_swa = window * (parallel + 1) if swa else ctx
    memory = n_full * per_full * ctx + n_swa * per_swa * cells_swa
    return {
        "layers_full": n_full,
        "layers_swa": n_swa,
        "sliding_window": swa or 0,
        "window_cells": window,
        "swa_cells": cells_swa if swa else 0,
        "bytes_per_token": read,
        "memory": memory,
        "per_full_token": per_full,
        "per_swa_token": per_swa,
    }


def kv_bytes_per_token(f: dict, cache_type: str, ctx: int = 0) -> float:
    """KV bytes read per generated token at the given context."""
    return kv_profile(f, cache_type, ctx or f.get("n_ctx_train", 0) or 4096)["bytes_per_token"]


def _cache_type_id(name: str) -> int:
    for tid, nm in GGML_NAMES.items():
        if nm.lower() == str(name).lower():
            return tid
    return 1  # f16


def ceilings(f: dict, bw_gbs: float | None, ctx: int, cache_type: str,
             peak_tflops: float | None, parallel: int = 1) -> dict:
    out = {"ctx": ctx, "cache_type": cache_type}
    prof = kv_profile(f, cache_type, ctx, parallel)
    out["kv_split"] = prof
    out["kv_read_per_token"] = prof["bytes_per_token"]
    out["kv_memory"] = prof["memory"]
    out["bytes_per_token_decode"] = f["active_bytes"] + prof["bytes_per_token"]

    p_active = f["n_params"]
    if f["n_expert"] and f["n_expert"] > 0:
        # approximate active params for MoE
        frac = (f["n_expert_used"] or f["n_expert"]) / f["n_expert"]
        p_active = int(f["n_params"] * frac)
    out["params_active"] = p_active
    out["flops_per_token"] = 2 * p_active

    if bw_gbs:
        out["decode_ceiling_tps"] = bw_gbs * 1e9 / out["bytes_per_token_decode"]
    else:
        out["decode_ceiling_tps"] = None

    if peak_tflops:
        out["prefill_ceiling_tps"] = peak_tflops * 1e12 / out["flops_per_token"]
    else:
        out["prefill_ceiling_tps"] = None

    out["model_arith_intensity"] = out["flops_per_token"] / f["active_bytes"]
    return out


def fit_accept(n_max: int, mean_len: float) -> float | None:
    """Recover the per-position acceptance rate from a measured mean length.

    The 'draft acceptance' ratio an engine logs is accepted/generated, which is
    not the per-position probability the geometric model needs.  Mean length is
    the right observable: it equals sum(a^i, i=0..n_max), which is monotonic in
    a, so bisect on it.
    """
    if not n_max or not mean_len or mean_len <= 1.0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        s = sum(mid ** i for i in range(n_max + 1))
        if s < mean_len:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def spec_prediction(f: dict, bw_gbs: float | None, ctx: int, cache_type: str,
                    accept: float, nmax_list: list, draft_bytes: int = 0) -> list:
    """Upper bound on decode throughput vs spec-draft-n-max.

    Charges the traffic a drafted token causes (the draft model's own weights
    for a separate draft model, or the target's output head for an inline MTP
    head) and the KV the target must read. It cannot model per-step latency or
    the draft model's own attention, both of which dominate at long context, so
    measurements beat this table wherever they exist.

    Per pass the target reads the weights once and drafts `n` tokens; each
    drafted token additionally costs a pass over the output head (to obtain
    draft logits) plus the MTP block.  Accepted length grows geometrically
    with the per-position acceptance rate.
    """
    if not bw_gbs:
        return []
    t_base = (f["active_bytes"]
              + kv_profile(f, cache_type, ctx)["bytes_per_token"]) / (bw_gbs * 1e9)
    # An inline MTP head reuses the target's output head, so that read is
    # charged per drafted token.  A *separate* draft model is its own small
    # network: charging the target head for it overstates the step by ~16x on
    # gemma-4 (680 MiB vs a 42 MiB assistant).
    if draft_bytes:
        draft_traffic = draft_bytes
    else:
        draft_traffic = f["output_head_bytes"] + f["mtp_bytes"]
    t_draft = draft_traffic / (bw_gbs * 1e9)
    rows = []
    for n in nmax_list:
        # expected accepted tokens = 1 + a + a^2 + ... + a^n
        exp_len = sum(accept ** i for i in range(n + 1))
        cost = t_base + n * t_draft
        rows.append({
            "n_max": n,
            "expected_len": round(exp_len, 2),
            "ms_per_pass": round(cost * 1000, 1),
            "predicted_tps": round(exp_len / cost, 2),
        })
    return rows


# --------------------------------------------------------------------------
# llama-server log parsing
# --------------------------------------------------------------------------

EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
ACCEPT_RE = re.compile(
    r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\),\s*"
    r"mean len\s*=\s*([\d.]+)")
# older builds print a per-position breakdown instead
DRAFT_OLD_RE = re.compile(
    r"draft-mtp:.*?#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+),\s*"
    r"#mean acc len\s*=\s*([\d.]+),\s*#acc rate/pos\s*=\s*\(([\d.]+)(?:\s*,\s*([\d.]+))?")
NGL_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")

COLS = ["name", "config", "prompt_tok", "prompt_tps", "out_tok", "tps",
        "ms_tok", "acc_len", "accept_rate", "passes", "ms_per_pass",
        "pow_w_max", "temp_c_max", "vram_pct_max"]


def _open_summary(path: str):
    """Append-mode handle, writing the column header only into an empty file.

    Must key off size, not existence: the orchestrator creates the file up
    front so that a run with zero successful configurations still leaves a
    readable summary.
    """
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    fh = open(path, "a")
    if need_header:
        fh.write("\t".join(COLS) + "\n")
    return fh


def _last(rx, text, group=None):
    ms = rx.findall(text)
    if not ms:
        return None
    return ms[-1]


def parse_run(log_path: str, power_path: str | None) -> dict:
    text = open(log_path, errors="replace").read()
    ev = _last(EVAL_RE, text)
    if not ev:
        return {}
    row = {c: "" for c in COLS}
    eval_ms, out_tok, ms_tok, tps = float(ev[0]), int(ev[1]), float(ev[2]), float(ev[3])
    row["out_tok"], row["tps"], row["ms_tok"] = out_tok, tps, ms_tok

    pr = _last(PROMPT_RE, text)
    if pr:
        row["prompt_tok"], row["prompt_tps"] = int(pr[1]), float(pr[3])

    ac = _last(ACCEPT_RE, text)
    if ac:
        rate, accepted, generated, mlen = float(ac[0]), int(ac[1]), int(ac[2]), float(ac[3])
        row["acc_len"], row["accept_rate"] = mlen, rate
        passes = round(out_tok / mlen) if mlen > 0 else 0
        row["passes"] = passes
        row["ms_per_pass"] = round(eval_ms / passes, 1) if passes else ""
    else:
        old = _last(DRAFT_OLD_RE, text)
        if old:
            row["acc_len"] = float(old[2])
            row["accept_rate"] = float(old[3])
            passes = round(out_tok / float(old[2])) if float(old[2]) > 0 else 0
            row["passes"] = passes
            row["ms_per_pass"] = round(eval_ms / passes, 1) if passes else ""

    if power_path and os.path.exists(power_path):
        try:
            rows = [l.split() for l in open(power_path) if l.strip()]
            vals = [(float(a), float(b), float(c)) for a, b, c in rows]
            if vals:
                row["pow_w_max"] = round(max(v[0] for v in vals), 1)
                row["temp_c_max"] = round(max(v[1] for v in vals), 1)
                row["vram_pct_max"] = round(max(v[2] for v in vals), 1)
        except (OSError, ValueError):
            pass
    return row


def _fmt(v, nd=2):
    if v in ("", None):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def is_auxiliary_gguf(path: str) -> bool:
    """True for companion GGUFs that are not the model itself.

    Repos ship draft/MTP assistants and mmproj files next to the model, and
    they often share the model's quant suffix -- e.g. ggml-org/gemma-4-E4B-it
    contains both `gemma-4-E4B-it-Q4_0.gguf` and
    `mtp-gemma-4-E4B-it-Q4_0.gguf`. Matching on the quant suffix alone picks
    the 60 MiB assistant instead of the 4.3 GiB model.
    """
    b = os.path.basename(path).lower()
    return (b.startswith(("mtp-", "mmproj", "draft-", "assistant-"))
            or "-mtp-" in b or "-mmproj" in b)


def resolve_hf(hf: str) -> str | None:
    """Map 'org/repo:QUANT' to a local GGUF path in the HF cache."""
    repo, _, quant = hf.partition(":")
    quant = quant or "Q4_K_M"
    base = os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + repo.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(base, slug, "snapshots", "*")))
    cands = []
    for s in snaps:
        cands += glob.glob(os.path.join(s, "*.gguf"))
    if not cands:
        return None
    main = [c for c in cands if not is_auxiliary_gguf(c)] or cands
    exact = [c for c in main if c.lower().endswith(f"-{quant.lower()}.gguf")]
    pool = exact or [c for c in main
                     if quant.lower() in os.path.basename(c).lower()] or main
    # the model is always the largest candidate in the pool
    return max(pool, key=os.path.getsize)


def find_companion_draft(model_path: str) -> str | None:
    """Locate a draft/assistant GGUF shipped beside the model.

    Some models carry their MTP head inline (nextn tensors); others ship it as a
    separate small model in the same directory, which needs
    `--spec-type draft-simple --spec-draft-model` rather than `draft-mtp`.
    """
    d = os.path.dirname(model_path)
    for pat in ("mtp-*.gguf", "*assistant*.gguf", "*draft*.gguf"):
        found = sorted(glob.glob(os.path.join(d, pat)))
        if found:
            return found[0]
    return None


def cmd_probe(a) -> int:
    info = {"binary": a.binary, "bench": a.bench, "requested_dev": a.dev}
    devs = enumerate_devices(a.bench, a.binary)
    info["devices"] = devs["devices"]
    info["backends"] = devs["backends"]
    info["uma"] = devs["uma"]

    if not devs["devices"]:
        print("ERROR: no devices reported by the build.", file=sys.stderr)
        return 2
    dev = None
    if a.dev:
        dev = next((d for d in devs["devices"] if d["id"].lower() == a.dev.lower()), None)
        if not dev:
            print(f"ERROR: device '{a.dev}' not available. Build supports: "
                  + ", ".join(d["id"] for d in devs["devices"]), file=sys.stderr)
            print("       (this build ships backends: " + ", ".join(devs["backends"]) + ")",
                  file=sys.stderr)
            return 2
    else:
        dev = devs["devices"][0]
        info["requested_dev"] = dev["id"]
    # detect_bandwidth needs the uma flag to choose the system-RAM path
    dev["uma"] = devs["uma"]
    info["device"] = dev

    if a.model:
        path = a.model
    else:
        path = resolve_hf(a.hf)
        if not path and a.download:
            bench = os.path.join(os.path.dirname(os.path.abspath(a.binary)), "llama-bench")
            if not os.path.exists(bench):
                bench = "llama-bench"
            print(f"model not cached; fetching {a.hf} via {bench} ...", file=sys.stderr)
            run([bench, "-hf", a.hf, "-p", "8", "-n", "1", "-r", "1"])
            path = resolve_hf(a.hf)
    if not path or not os.path.exists(path):
        print(f"ERROR: could not locate a GGUF for {a.hf or a.model}. "
              f"Pass --model PATH or --download.", file=sys.stderr)
        return 2
    info["model_path"] = path

    facts = model_facts(path)
    info["facts"] = facts

    # Requested context is clamped to the model's trained maximum: asking for
    # more than a model supports wastes KV memory and is silently capped by the
    # engine anyway, so the harness reports and uses the effective value.
    ctx_req = a.ctx
    n_ctx_train = facts.get("n_ctx_train") or 0
    ctx = min(ctx_req, n_ctx_train) if n_ctx_train else ctx_req
    info["ctx_requested"] = ctx_req
    info["ctx_effective"] = ctx

    # Speculative decoding needs a draft source. Some models embed an MTP head
    # (nextn tensors); others ship it as a separate small GGUF beside the model,
    # which must be passed as --spec-draft-model. The engine's draft-mtp type
    # drives either one -- verified on ggml-org/gemma-4-E4B-it, where draft-mtp
    # plus the companion file gives 1.9x decode, while draft-simple on the same
    # pair fails outright.
    if a.draft_model == "none":
        draft = None
    elif a.draft_model:
        draft = a.draft_model if os.path.exists(a.draft_model) else None
    elif facts.get("has_mtp"):
        draft = None                      # head is inline, nothing to load
    else:
        draft = find_companion_draft(path)
    info["draft_model"] = draft or ""
    draft_bytes = 0
    if draft and os.path.exists(draft):
        try:
            draft_bytes = model_facts(draft).get("tensor_bytes", 0)
        except (OSError, ValueError):
            draft_bytes = os.path.getsize(draft)
    info["draft_bytes"] = draft_bytes

    # A repo may also ship a multimodal projector, which the engine loads
    # automatically when the model is resolved by HF repo id (--mmproj-auto
    # defaults on). It is dead weight for text-only serving: measured 0.52 GiB
    # of memory, but no measurable throughput cost (21.72 vs 21.63 t/s).
    mmproj = None
    for cand in sorted(glob.glob(os.path.join(os.path.dirname(path), "mmproj*.gguf"))):
        mmproj = cand
        break
    info["mmproj"] = mmproj or ""
    info["mmproj_bytes"] = os.path.getsize(mmproj) if mmproj else 0
    info["draft_origin"] = ("override" if a.draft_model and a.draft_model != "none"
                            else "inline" if facts.get("has_mtp")
                            else "companion" if draft else "none")

    bw = detect_bandwidth(dev, {"bus_bits": a.mem_bus_bits, "mt_s": a.mem_mt_s})
    info["bandwidth"] = bw
    bw_gbs = bw["theoretical_gbs"]

    ceil = ceilings(facts, bw_gbs, ctx, a.cache_type, a.peak_tflops, a.parallel)
    info["ceilings"] = ceil

    nmax_list = [2, 4, 6, 8, 12]
    info["spec_prediction"] = spec_prediction(
        facts, bw_gbs, ctx, a.cache_type, a.assumed_accept, nmax_list,
        draft_bytes=draft_bytes)
    info["assumed_accept"] = a.assumed_accept

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "probe.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)

    # ---- print -----------------------------------------------------------
    def h(t):
        print(f"\n{t}\n" + "-" * len(t))

    print(f"llmfit probe   binary={a.binary}")
    print(f"  build backends : {', '.join(devs['backends'])}")
    for d in devs["devices"]:
        mark = "  <-- using" if d["id"] == dev["id"] else ""
        print(f"  device {d['id']:9}: {d['name']}  {d['total_mib']} MiB "
              f"({d['free_mib']} MiB free){mark}")
    print(f"  unified memory : {devs['uma']}")

    h("model")
    f = facts
    print(f"  path           : {path}")
    print(f"  arch / name    : {f['arch']} / {f['name']} {f['size_label']}")
    print(f"  file size      : {f['file_bytes']/2**30:.2f} GiB")
    print(f"  params         : {f['n_params']/1e9:.2f} B"
          + (f"  (declared {f['n_params_declared']/1e9:.2f} B)"
             if isinstance(f['n_params_declared'], int) else ""))
    if f["n_expert"]:
        print(f"  MoE experts    : {f['n_expert_used']}/{f['n_expert']} per token"
              f"  -> active weights {f['active_bytes']/2**30:.2f} GiB")
    print(f"  layers/embd/ff : {f['n_layer']} / {f['n_embd']} / {f['n_ff']}")
    print(f"  heads (kv)     : {f['n_head']} ({f['n_head_kv']}), k/v len "
          f"{f['k_len']}/{f['v_len']}, vocab {f['n_vocab']}")
    print(f"  MTP head       : {'yes, block ' + str(f['mtp_block']) if f['has_mtp'] else 'no'}"
          + (f"  ({f['mtp_bytes']/2**20:.0f} MiB)" if f["has_mtp"] else ""))
    if info.get("mmproj"):
        print(f"  mmproj         : {os.path.basename(info['mmproj'])}"
              f"  ({info['mmproj_bytes']/2**20:.0f} MiB)")
        print("                   auto-loaded when resolved by HF repo id; costs")
        print("                   memory but not text throughput. Disable with")
        print("                   --no-mmproj-auto for text-only serving.")
    if info["draft_model"]:
        print(f"  draft model    : {info['draft_model']}  ({info['draft_origin']})")
    elif not f["has_mtp"]:
        print("  draft model    : none found - speculative decoding cannot draft for")
        print("                   this model; pass --draft-model PATH if one exists")
    print(f"  output head    : {f['output_head_bytes']/2**20:.0f} MiB "
          f"({f['output_head_type']})")
    if ctx != ctx_req:
        print(f"  context        : {ctx_req} requested -> {ctx} "
              f"(model maximum)")
    need_mib = (f["active_bytes"] + ceil.get("kv_memory", 0)) / 2**20
    free_mib = dev.get("free_mib")
    if free_mib and need_mib > free_mib:
        print(f"  WARNING        : weights + KV need ~{need_mib:.0f} MiB but only"
              f" {free_mib} MiB is free; lower --ctx or offload")
    if f["unknown_types"]:
        print(f"  WARNING        : unknown tensor types {f['unknown_types']}, "
              f"size estimate unreliable")
    drift = abs(f["tensor_bytes"] - f["file_bytes"]) / max(f["file_bytes"], 1)
    if drift > 0.06:
        print(f"  WARNING        : tensor sum {f['tensor_bytes']/2**30:.2f} GiB vs file "
              f"{f['file_bytes']/2**30:.2f} GiB ({drift*100:.0f}% off)")

    h("memory bandwidth")
    for d in bw["detail"]:
        print(f"  {d}")
    if bw_gbs:
        print(f"  theoretical    : {bw_gbs:.1f} GB/s   [{bw['source']}]")
    else:
        print(f"  theoretical    : unknown   [{bw['source']}]")

    h(f"ceilings at ctx={ctx} ({a.cache_type} KV)")
    ks = ceil.get("kv_split", {})
    print(f"  KV read per token   : {ceil.get('kv_read_per_token',0)/2**20:.2f} MiB"
          f"   (at ctx={ctx}, so depends on ctx)")
    print(f"  KV memory at ctx    : {ceil.get('kv_memory',0)/2**20:.1f} MiB"
          f"   ({ks.get('layers_full',0)} full-attention + "
          f"{ks.get('layers_swa',0)} windowed layers"
          + (f", window {ks.get('sliding_window')}" if ks.get("sliding_window") else "")
          + ")")
    print(f"  bytes/token decode  : {ceil['bytes_per_token_decode']/2**30:.2f} GiB")
    print(f"  FLOPs/token prefill : {ceil['flops_per_token']/1e9:.1f} GFLOP")
    print(f"  model intensity     : {ceil['model_arith_intensity']:.1f} FLOP/byte")
    if ceil["decode_ceiling_tps"]:
        print(f"  decode ceiling      : {ceil['decode_ceiling_tps']:.2f} t/s "
              f"(bandwidth-bound, no speculation)")
    if ceil["prefill_ceiling_tps"]:
        print(f"  prefill ceiling     : {ceil['prefill_ceiling_tps']:.0f} t/s "
              f"(compute-bound, at {a.peak_tflops} TFLOPS peak)")
    else:
        print(f"  prefill ceiling     : pass --peak-tflops for an efficiency figure")

    h("structure")
    nh = facts.get("n_attn_layers", 0)
    nr = facts.get("n_recurrent_layers", 0)
    nkv = facts.get("n_kv_layers", f["n_layer"])
    kind = ("hybrid" if nr and nh else
            "MoE" if f["n_expert"] else "dense")
    print(f"  kind            : {kind}"
          + (f"  ({nh} full-attention, {nr} recurrent/SSM)" if nr and nh else ""))
    print(f"  blocks          : {facts.get('n_blocks', f['n_layer'])}")
    print(f"  KV-bearing      : {nkv} of {facts.get('n_blocks', f['n_layer'])} layers"
          f"   -> {ceil.get('kv_read_per_token',0)/2**20:.2f} MiB read/token at ctx")
    pat = facts.get("layer_pattern", "")
    if pat:
        limit = 72
        shown = pat if len(pat) <= limit else pat[:limit] + f"... (+{len(pat)-limit})"
        print(f"  layer pattern   : {shown}   (A=KV layer, R=recurrent)")
    if nr:
        print(f"  recurrent state : context-independent (SSM layers keep a fixed"
              f" state, not a growing KV)")
    th = facts.get("type_hist", {})
    if th:
        top = ", ".join(f"{k}x{v}" for k, v in list(th.items())[:7])
        print(f"  quant mix       : {top}")

    print("  bytes by component:")
    for comp, b in (facts.get("component_bytes") or {}).items():
        pct = 100 * b / max(f["tensor_bytes"], 1)
        note = ""
        if comp == "experts" and f["n_expert"]:
            note = (f"   <- only {f['n_expert_used']}/{f['n_expert']} routed per token"
                    f" = {100*f['n_expert_used']/f['n_expert']:.0f}% read")
        elif comp == "output head":
            note = "   <- read once per drafted token during speculation"
        print(f"    {comp:<16} {b/2**30:7.3f} GiB  {pct:5.1f}%{note}")

    print("  layer archetypes (blocks sharing an identical tensor set):")
    for i, a_ in enumerate(facts.get("archetypes", [])[:4]):
        idx = a_["indices"]
        shown = ", ".join(str(x) for x in idx[:6]) + ("..." if len(idx) > 6 else "")
        print(f"    {a_['count']:>3} x  [{shown}]   {a_['bytes']/2**30:.3f} GiB")
        print(f"          {', '.join(a_['roles'][:5])}")
        if len(a_["roles"]) > 5:
            print(f"          ... +{len(a_['roles'])-5} more")

    if info["spec_prediction"]:
        h(f"predicted spec-decoding (assumed per-position acceptance "
          f"{a.assumed_accept:.2f})")
        print("  UPPER BOUND only: this charges per-step traffic but cannot know")
        print("  per-step latency or draft attention, which dominate in practice.")
        print("  The fitted table in the report supersedes it.")
        print(f"  {'n_max':>5} {'exp.len':>8} {'ms/pass':>9} {'t/s':>8}")
        for r in info["spec_prediction"]:
            print(f"  {r['n_max']:>5} {r['expected_len']:>8} "
                  f"{r['ms_per_pass']:>9} {r['predicted_tps']:>8}")
        best = max(info["spec_prediction"], key=lambda r: r["predicted_tps"])
        print(f"  -> predicted optimum n_max={best['n_max']} "
              f"({best['predicted_tps']} t/s, {best['predicted_tps']/max(ceil['decode_ceiling_tps'] or 1,1e-9):.1f}x "
              f"the unspeculated ceiling)")
    print()
    return 0


def cmd_parse(a) -> int:
    row = parse_run(a.log, a.power)
    if not row:
        print(f"{a.name}: no 'eval time' line in {a.log} (incomplete run?)", file=sys.stderr)
        return 1
    row["name"] = a.name
    row["config"] = a.config or ""
    with _open_summary(a.summary) as fh:
        fh.write("\t".join(_fmt(row[c]) for c in COLS) + "\n")
    print(f"  {a.name:22} prefill={_fmt(row['prompt_tps']):>7} t/s  "
          f"decode={_fmt(row['tps']):>6} t/s  acc_len={_fmt(row['acc_len'])}  "
          f"accept={_fmt(row['accept_rate'])}")
    return 0


def _blocks_to_text(content, include_thinking: bool) -> str:
    """Flatten an OMP message content list into text."""
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text" or "text" in b:
            out.append(b.get("text") or "")
        elif kind == "thinking" or "thinking" in b:
            if include_thinking:
                out.append(b.get("thinking") or "")
    return "\n".join(x for x in out if x)


def read_session(path: str, include_thinking: bool = False) -> dict:
    """Read an OMP session recording (JSONL) into ordered messages.

    OMP writes one JSON object per line under
    ~/.omp/agent/sessions/<project>/<ts>_<id>.jsonl.  Records are typed:
    "session" (header), "model_change", "title", "custom", and "message", where
    message.message carries role plus a content list of typed blocks.  Assistant
    records also carry usage/ttft/duration, i.e. the observed performance of the
    real workload.
    """
    header, messages, models = {}, [], []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        t = o.get("type")
        if t == "session":
            header = o
        elif t == "model_change" and o.get("model"):
            models.append(o["model"])
        elif t == "message":
            m = o.get("message", {})
            role = m.get("role")
            text = _blocks_to_text(m.get("content"), include_thinking)
            if role and text:
                messages.append({"role": role, "text": text,
                                 "usage": m.get("usage"), "model": m.get("model"),
                                 "ttft": m.get("ttft"), "duration": m.get("duration")})
    return {"header": header, "messages": messages, "models": models}


def build_turns(session: dict, max_chars: int) -> list:
    """Turn a recorded conversation into a sequence of cumulative prompts.

    Each *user* turn becomes one prompt containing everything up to that point,
    which reproduces the real serving pattern: context grows as the session
    proceeds, so prefill and decode are measured at the depths actually seen.
    """
    turns, acc = [], []
    for m in session["messages"]:
        # A prompt is emitted at each model call, i.e. before every assistant
        # message: that is exactly the text the server received, so its length
        # is the real context depth.  Emitting only at user turns misses the
        # tool results, which dominate a coding session's context.
        if m["role"] == "assistant":
            prompt = "\n\n".join(acc)
            if prompt:
                if max_chars and len(prompt) > max_chars:
                    prompt = prompt[-max_chars:]   # keep the recent context
                turns.append({"index": len(turns), "chars": len(prompt),
                              "prompt": prompt})
        acc.append(f"[{m['role']}]\n{m['text']}")
    # an unterminated tail (session ended on a user/tool message) is still a
    # real request that was about to be made
    if acc:
        prompt = "\n\n".join(acc)
        if max_chars and len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        turns.append({"index": len(turns), "chars": len(prompt), "prompt": prompt})
    return turns


def _post_completion(port: int, prompt: str, gen: int, seed: int) -> None:
    """POST one completion to the local server. Uses a raw socket write so the
    harness stays dependency-free."""
    import socket
    body = json.dumps({
        "prompt": prompt, "max_tokens": gen,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repeat_penalty": 1.0,
        "seed": seed, "stream": False, "stop": [], "ignore_eos": True,
    }).encode()
    head = (f"POST /completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=21600) as s:
        s.sendall(head + body)
        while s.recv(1 << 16):
            pass


def cmd_replay_run(a) -> int:
    """Post each extracted turn to the server, in order, on one slot.

    Sequential requests on a single slot make the server reuse the shared
    prefix, so the context grows turn by turn exactly as it did live.
    """
    rows = [json.loads(l) for l in open(a.turns) if l.strip()]
    if a.limit and a.limit > 0:
        rows = rows[:a.limit]
    ok = 0
    for i, t in enumerate(rows):
        try:
            _post_completion(a.port, t["prompt"], a.gen, a.seed)
            ok += 1
            print(f"  turn {i}: {t['chars']} chars", file=sys.stderr)
        except OSError as e:
            print(f"  turn {i}: failed: {e}", file=sys.stderr)
    print(f"  completed {ok}/{len(rows)} turns", file=sys.stderr)
    return 0 if ok else 1


def cmd_replay_report(a) -> int:
    """Per-turn table for a replayed session log.

    One server run serves every turn, so the log holds one print_timing block
    per task; group by task id and pair each prompt-eval with its eval and
    acceptance line.
    """
    task_re = re.compile(r"print_timing:.*?task\s+(\d+)\s*\|(.*)$")
    per = {}
    order = []
    for line in open(a.log, errors="replace"):
        m = task_re.search(line)
        if not m:
            continue
        tid, rest = int(m.group(1)), m.group(2)
        if tid not in per:
            per[tid] = {}
            order.append(tid)
        d = per[tid]
        p = PROMPT_RE.search(rest)
        e = EVAL_RE.search(rest)
        c = ACCEPT_RE.search(rest)
        if p:
            d["ptok"], d["ptps"] = int(p.group(2)), float(p.group(4))
        if e:
            d["otok"], d["tps"] = int(e.group(2)), float(e.group(4))
        if c:
            d["mlen"] = float(c.group(4))
            d["acc"] = float(c.group(1))

    rows = [(t, per[t]) for t in order if "tps" in per[t]]
    if not rows:
        print("no completed turns in the log", file=sys.stderr)
        return 1
    print(f"\n  replayed {len(rows)} turns"
          + (f" (first {a.max_turns} of {len(order)})" if a.max_turns else ""))
    print(f"  {'turn':>4} {'prompt tok':>11} {'prefill t/s':>12} {'out tok':>8}"
          f" {'decode t/s':>11} {'mean len':>9}")
    tot_pt = tot_ot = 0
    ptps = []
    for i, (tid, d) in enumerate(rows):
        tot_pt += d.get("ptok", 0)
        tot_ot += d.get("otok", 0)
        ptps.append(d.get("ptps", 0.0))
        print(f"  {i:>4} {d.get('ptok',0):>11} {d.get('ptps',0):>12.1f}"
              f" {d.get('otok',0):>8} {d.get('tps',0):>11.2f}"
              f" {d.get('mlen',0):>9.2f}")
    import statistics
    print(f"\\n  totals: {tot_pt} prompt tokens, {tot_ot} generated tokens")
    print("  prompt tok counts only NEW tokens: the server reuses the shared")
    print("  prefix across turns, so turn 1 onward processes far fewer than their")
    print("  full prompt size -- that reuse is what the real session experienced.")
    print(f"  prefill: median {statistics.median(ptps):.1f} t/s"
          f"  min {min(ptps):.1f}  max {max(ptps):.1f}")
    dtps = [d["tps"] for _, d in rows]
    print(f"  decode : median {statistics.median(dtps):.2f} t/s"
          f"  min {min(dtps):.2f}  max {max(dtps):.2f}")
    ml = [d["mlen"] for _, d in rows if "mlen" in d]
    if ml:
        print(f"  accepted length: median {statistics.median(ml):.2f}"
              f"  (measured on THIS workload, unlike the synthetic sweep prompt)")
    return 0


def cmd_replay_extract(a) -> int:
    sess = read_session(a.session, a.include_thinking)
    if not sess["messages"]:
        print(f"no messages found in {a.session}", file=sys.stderr)
        return 1
    turns = build_turns(sess, a.max_chars)
    if not turns:
        print("no user turns found", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    tp = os.path.join(a.out, "turns.jsonl")
    with open(tp, "w") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")
    longest = max(turns, key=lambda t: t["chars"])
    pp = os.path.join(a.out, "prompt.txt")
    open(pp, "w").write(longest["prompt"])

    # Infer which model served the session so --replay needs no --hf/--model.
    # A recording may name several: a bare alias ("local-llama/gemma-4-E4B-it-Q4_0")
    # as well as a resolvable one ("local-llama/<repo>:<quant>") or a local path
    # ("llama.cpp//abs/path.gguf"). Emit every candidate best-first and let the
    # caller pick the first that actually resolves, rather than trusting order.
    hints = []
    for m in sess["models"]:
        for tok in m.split():
            if tok.endswith(".gguf") and "/" in tok:
                hints.append(tok if tok.startswith("/") else "/" + tok.lstrip("/"))
        if m.startswith("local-llama/"):
            rest = m.split("/", 1)[1]
            if "/" in rest and ":" in rest:
                hints.append("hf:" + rest)
    seen = set()
    hints = [h for h in hints if not (h in seen or seen.add(h))]
    if hints:
        with open(os.path.join(a.out, "model.txt"), "w") as fh:
            fh.write("\n".join(hints) + "\n")

    h = sess["header"]
    print(f"session : {os.path.basename(a.session)}")
    print(f"  cwd   : {h.get('cwd','?')}")
    print(f"  title : {h.get('title','?')}")
    if sess["models"]:
        print(f"  models: {', '.join(dict.fromkeys(sess['models']))}")
    roles = {}
    for m in sess["messages"]:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
    print(f"  records: {len(sess['messages'])}  " +
          "  ".join(f"{k}={v}" for k, v in roles.items()))
    print(f"  user turns: {len(turns)}")
    print(f"  prompt size: min {min(t['chars'] for t in turns)}"
          f"  max {max(t['chars'] for t in turns)} chars"
          f"  (~{max(t['chars'] for t in turns)//4} tokens)")
    print(f"  wrote : {tp}")
    print(f"          {pp}   (largest turn, for --prompt)")
    if a.show_usage:
        obs = [m for m in sess["messages"] if m.get("usage")]
        if obs:
            pt = [m["usage"].get("prompt_tokens") or m["usage"].get("input_tokens")
                  for m in obs]
            pt = [x for x in pt if isinstance(x, int)]
            ct = [m["usage"].get("completion_tokens") or m["usage"].get("output_tokens")
                  for m in obs]
            ct = [x for x in ct if isinstance(x, int)]
            tt = [m["ttft"] for m in obs if isinstance(m.get("ttft"), (int, float))]
            if pt:
                print(f"  recorded: prompt_tokens {min(pt)}..{max(pt)}"
                      f" (median {sorted(pt)[len(pt)//2]})")
            if ct:
                print(f"            completion_tokens {min(ct)}..{max(ct)}"
                      f" (median {sorted(ct)[len(ct)//2]})")
            if tt:
                print(f"            ttft {min(tt):.0f}..{max(tt):.0f} ms")
    return 0


def cmd_raw(a) -> int:
    """Turn llama-bench output into summary rows (pp512 / tg128).

    llama-bench prints a markdown table whose last two columns are the test
    name and the throughput, so parse positionally rather than by column count
    (the column set varies with which options were swept).
    """
    text = open(a.bench_log, errors="replace").read()
    n = 0
    with _open_summary(a.summary) as fh:
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            test, tps = cells[-2], cells[-1].split("±")[0].strip()
            # llama-bench appends the depth to the test name ("tg128 @ d8192")
            m = re.fullmatch(r"(pp|tg)(\d+)(?:\s*@\s*d(\d+))?", test)
            if not m:
                continue
            try:
                val = float(tps)
            except ValueError:
                continue
            kind, ntok, depth = m.group(1), m.group(2), int(m.group(3) or 0)
            row = {c: "" for c in COLS}
            row["name"] = (f"{a.prefix}{kind}{ntok}"
                           + (f"_d{depth}" if depth else ""))
            test = kind + ntok
            row["config"] = a.config or ""
            if test.startswith("pp"):
                row["prompt_tok"] = test[2:]
                row["prompt_tps"] = val
            else:
                row["out_tok"] = test[2:]
            row["tps"] = val
            fh.write("\t".join(_fmt(row[c]) for c in COLS) + "\n")
            n += 1
            print(f"  {row['name']:22} {val:.2f} t/s")
    return 0 if n else 1


def _load_rows(outdir: str) -> list:
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "**", "summary.tsv"), recursive=True)):
        lines = open(f).read().splitlines()
        if len(lines) < 2:
            continue
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) == len(hdr):
                rows.append(dict(zip(hdr, p)))
    return rows


def fit_pass_cost(points) -> tuple | None:
    """Fit pass cost = t_base + n * t_draft from measured (n, ms_per_pass).

    A single least-squares line is wrong here: on this machine the marginal cost
    of a drafted step is ~4.6 ms up to n=6 and then jumps to ~36 ms at n=8. A
    line fitted through both regimes matches neither. Marginal cost per step is
    therefore inspected first, and points beyond a step that is 3x the median
    are dropped from the fit and reported as a cliff -- the model then describes
    the region it can actually describe.
    """
    pts = sorted(points)
    if len(pts) < 2:
        return None
    margs = [((c2 - c1) / (n2 - n1), n2)
             for (n1, c1), (n2, c2) in zip(pts, pts[1:]) if n2 > n1]
    if not margs:
        return None
    med = statistics.median([m for m, _ in margs])
    cliff, keep = None, list(pts)
    for m, n2 in margs:
        if med > 0 and m > 3 * med:
            cliff, keep = n2, [p for p in pts if p[0] < n2]
            break
    if len(keep) < 2:
        keep = pts[:2]
    ns = [p[0] for p in keep]
    cs = [p[1] for p in keep]
    k = len(ns)
    sn, sc = sum(ns), sum(cs)
    snn = sum(n * n for n in ns)
    snc = sum(n * c for n, c in zip(ns, cs))
    den = k * snn - sn * sn
    if abs(den) < 1e-9:
        return None
    t_draft = (k * snc - sn * sc) / den
    t_base = (sc - t_draft * sn) / k
    return max(t_base, 0.0), max(t_draft, 0.0), cliff


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def structure_report(facts: dict, probe: dict, ctx: int, cache_type: str,
                     parallel: int = 1) -> None:
    """Structure breakdown, memory budget and offload guidance.

    The point of the byte split is to answer "would keeping some layers off the
    GPU help?".  For a dense or hybrid stack the answer is no: every weight is
    read on every decode step, so moving a layer to CPU removes no GPU traffic
    and only adds a slower path.  The exceptions are weights that are genuinely
    not read every token -- inactive MoE experts, and the MTP block when
    speculation is off -- and those are what this section quantifies.
    """
    f = facts
    kb = f.get("kv", {})
    arch = f.get("arch", "")

    def h(t):
        print(f"\n{t}\n" + "-" * len(t))

    h("structure")
    nh, nr = f.get("n_attn_layers", 0), f.get("n_recurrent_layers", 0)
    kind = "hybrid" if (nr and nh) else ("MoE" if f.get("n_expert") else "dense")
    print(f"  kind          : {kind}"
          + (f"   ({nh} attention, {nr} recurrent/SSM)" if nr and nh else ""))
    nb = f.get("n_blocks", f.get("n_layer", 0))
    print(f"  blocks        : {nb}")
    kp = kv_profile(f, cache_type, ctx, parallel)
    print(f"  KV-bearing    : {f.get('n_kv_layers','?')} of {nb} layers"
          f"  ({kp['layers_full']} full-attention, {kp['layers_swa']} windowed at"
          f" {cache_type})")
    pat = f.get("layer_pattern", "")
    if pat and len(set(pat)) > 1:
        limit = 96
        print(f"  layer pattern : "
              + (pat if len(pat) <= limit else pat[:limit] + f"... (+{len(pat)-limit})"))
        print("                  A=holds KV   R=recurrent   M=MTP block")

    th = f.get("type_hist", {})
    if th:
        print(f"  quant mix     : " + ", ".join(f"{k}x{v}" for k, v in list(th.items())[:7]))

    if f.get("tied_head"):
        print(f"  LM head       : tied to token_embd "
              f"({f.get('output_head_bytes', 0)/2**20:.0f} MiB, counted under"
              f" embeddings below)")
    print("  bytes by component:")
    for comp, b in (f.get("component_bytes") or {}).items():
        pct = 100 * b / max(f.get("tensor_bytes", 1), 1)
        extra = ""
        if comp == "experts" and f.get("n_expert"):
            extra = (f"   <- {f['n_expert_used']}/{f['n_expert']} routed per token")
        elif comp == "mtp head":
            extra = "   <- read only when speculation is on"
        print(f"    {comp:<16} {b/2**30:7.3f} GiB  {pct:5.1f}%{extra}")

    # ---- memory budget ---------------------------------------------------
    h(f"memory budget at ctx={ctx}")
    weights = f.get("tensor_bytes", 0)
    kv_b = kv_profile(f, cache_type, ctx, parallel)["memory"]
    free_mib = probe.get("device", {}).get("free_mib")
    print(f"  {'weights':<26}{weights/2**30:7.2f} GiB")
    if kp["layers_swa"]:
        kv_desc = (f"{kp['layers_full']} full x {ctx} cells, "
                   f"{kp['layers_swa']} windowed x {kp['swa_cells']}")
    else:
        kv_desc = f"{f.get('n_kv_layers','?')} layers x {ctx} cells"
    print(f"  {'KV at ctx (' + cache_type + ')':<26}{kv_b/2**30:7.2f} GiB"
          f"   ({kv_desc})")
    print(f"  {'compute buffers':<26}{0.2:7.2f} GiB   approx, grows with batch")
    print(f"  {'-'*40}")
    print(f"  {'subtotal':<26}{(weights+kv_b+0.2*2**30)/2**30:7.2f} GiB")
    if free_mib:
        print(f"  {'device free':<26}{free_mib/1024:7.2f} GiB")
    if kb.get(f"{arch}.ssm.state_size"):
        print("  plus a recurrent-state buffer that the SSM layers keep instead of a")
        print("  growing cache: its size does not depend on --ctx, so raising the")
        print("  context is cheap here. exact figure: llama-server -lv 4")

    # ---- offload guidance ------------------------------------------------
    h("offload guidance")
    if f.get("n_expert"):
        ex = f.get("expert_bytes", 0)
        nx = f.get("nonexpert_bytes", 0)
        used, tot = f.get("n_expert_used", 1), f.get("n_expert", 1)
        frac = used / tot if tot else 1
        print(f"  MoE: expert weights are {ex/2**30:.2f} GiB of {weights/2**30:.2f} GiB"
              f" ({100*ex/max(weights,1):.0f}%), but only")
        print(f"  {used}/{tot} experts are routed per token, so per-token expert"
              f" traffic is {ex*frac/2**30:.2f} GiB.")
        print(f"  Keeping experts in system RAM (--cpu-moe, or -ncmoe N for the"
              f" first N layers)")
        print(f"  therefore removes weight from the GPU at a cost proportional to"
              f" the ROUTED")
        print(f"  fraction ({frac:.2f}), not the total. This is the first thing to"
              f" offload when")
        print(f"  the model does not fit: GPU-resident per-token traffic falls from"
              f" ~{(f.get('active_bytes',0)+ex*0)/2**30:.2f}")
        print(f"  to {nx/2**30:.2f} GiB, at the price of CPU-side expert compute.")
    else:
        print("  dense: every weight is read on every decode step, so moving any")
        print("  layer to the CPU removes no GPU traffic - it only adds a slower")
        print("  path for that layer's share. Partial offload is for FITTING, not")
        print("  for speed. The only bytes that can be skipped entirely are the")
        print("  MTP head (when speculation is off) and, in an MoE, unrouted experts.")
    if nr:
        print(f"  recurrent layers ({nr}): their advantage is that they need no KV,"
              f" not that")
        print("  they are cheap - their weights are still fully read every token.")
        print("  They are why this model's context is inexpensive, not a reason to")
        print("  move them off the GPU.")
    if f.get("n_kv_layers", 0) and f.get("n_kv_layers", 0) != nb:
        print(f"  only {f['n_kv_layers']} of {nb} layers hold KV, so --ctx sizing"
              f" follows the")
        print(f"  KV-bearing count ({f['n_kv_layers']}), not the block count ({nb}).")


def write_ini(path: str, probe: dict, ranked: list, best_key: str, best_mean: float,
              unspec: float | None, other: list, hf: str, outdir: str) -> str:
    """Write a llama.cpp model-preset INI from a finished run.

    Format reference: docs/preset.md and tools/server/README.md ("Model presets").
    Keys are command-line arguments without the leading dashes; short forms and
    LLAMA_ARG_* names are also accepted.  `[*]` holds settings shared by every
    model in the file, a named section holds this model's tuned options and
    merges into the matching cached model entry when the name is an HF id.
    """
    facts = probe.get("facts", {})
    dev = probe.get("device", {})
    ceil = probe.get("ceilings", {})
    bw = probe.get("bandwidth", {})

    # the winning key is a space-separated "k=v" list; read each axis from it so a
    # KV-cache or ubatch win is reflected in the preset too
    bk = best_key or ""

    def _key(rx: str, default: str) -> str:
        m = re.search(rx, bk)
        return m.group(1) if m else default

    nmax = _key(r"n-max=(\d+)", "4")
    pmin = _key(r"p-min=([\d.]+)", "0.0")
    kv = _key(r"kv=(\S+)", ceil.get("cache_type", "q4_0"))
    ub = _key(r"ub=(\d+)", "512")
    b = _key(r"b=(\d+)", "1024")
    fa = "off" if "fa=off" in bk else "on"
    spec_type = "draft-mtp,ngram-mod" if "ngram" in bk else "draft-mtp"

    ctx = ceil.get("ctx", 32768)
    abspath = os.path.abspath(path)
    # A section name equal to the HF id merges into that cached model entry;
    # otherwise the section needs `model` (or `hf-repo`) to resolve.
    if hf:
        section = hf
        model_line = None
    else:
        model_line = facts.get("path", "")
        section = os.path.splitext(os.path.basename(model_line))[0] or "model"

    L = []
    L.append("; llama-models-options.ini")
    L.append(f"; generated by llmfit on {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    L.append(f"; from run: {outdir}")
    L.append(";")
    L.append(f"; model  : {facts.get('arch','?')} {facts.get('name','')} "
             f"{facts.get('size_label','')} "
             f"({facts.get('file_bytes',0)/2**30:.2f} GiB, "
             f"{facts.get('n_params',0)/1e9:.2f} B params)")
    L.append(f"; device : {dev.get('id','?')} {dev.get('name','')}")
    L.append(";")
    L.append("; measured (this run, single stream):")
    L.append(f";   context used for measurements: {ctx}")
    if ranked:
        L.append(f";   best decode config {'of ' + str(max(t[2] for t in ranked)) + ' round(s)' if ranked else ''}"
                 f": {best_key} -> {best_mean:.2f} t/s")
    if unspec:
        L.append(f";   without speculation        : {unspec:.2f} t/s"
                 f"  ({best_mean/unspec:.2f}x with it)" if best_mean else "")
    theo = bw.get("theoretical_gbs")
    if theo:
        L.append(f";   memory bandwidth           : {theo:.1f} GB/s theoretical")
    if ceil.get("decode_ceiling_tps"):
        L.append(f";   decode ceiling             : "
                 f"{ceil['decode_ceiling_tps']:.2f} t/s (bandwidth-bound)")
        L.append(";   -> decode is close to the bandwidth ceiling, so speculative")
        L.append(";      decoding is the only lever left on this hardware.")
    if ceil.get("prefill_ceiling_tps"):
        L.append(f";   prefill ceiling            : "
                 f"{ceil['prefill_ceiling_tps']:.0f} t/s (compute-bound)")
    L.append(";")
    for r in other or []:
        L.append(f"; other workload : {r.get('config','')} -> {r.get('tps','')} t/s")
    L.append(";")
    L.append("; Use with:  llama-server --models-preset " + abspath)
    L.append(";   then request the model by name, or run llama-server -hf " + (hf or section))
    L.append("; Note: host, port, api-key, alias and hf-repo are router-controlled")
    L.append("; and are removed or overwritten when a preset is loaded.")
    L.append("")
    L.append("version = 1")
    L.append("")
    L.append("; Device/backend-wide settings. CAUTION: a [*] section also applies to")
    L.append("; every other model loaded through this preset file, so it deliberately")
    L.append("; holds only values that are correct regardless of model. Anything that")
    L.append("; depends on the model's size, context or workload lives below.")
    L.append("[*]")
    L.append(f"device       = {dev.get('id','')}")
    L.append("flash-attn   = " + ("on" if fa == "on" else "off"))
    L.append(f"cache-type-k = {kv}")
    L.append(f"cache-type-v = {kv}")
    L.append("")
    L.append("; Model-specific configuration. The tuned group is speculative decoding;")
    L.append("; the rest reproduces the configuration the measurement used.")
    L.append(f"[{section}]")
    L.append(f"ctx-size     = {ctx}")
    L.append(f"batch-size   = {b}")
    L.append(f"ubatch-size  = {ub}")
    L.append("parallel     = 1")
    if model_line:
        L.append(f"model = {model_line}")
    else:
        L.append(f"; resolves via the cached entry named above; add an explicit path")
        L.append(f"; instead of the section name if you move the file:")
        L.append(f"; model = {facts.get('path','')}")
    L.append(f"spec-type            = {spec_type}")
    L.append(f"spec-draft-n-max     = {nmax}")
    L.append(f"spec-draft-p-min     = {pmin}")
    L.append(f"spec-draft-type-k    = {kv}")
    L.append(f"spec-draft-type-v    = {kv}")
    L.append(f"spec-draft-device    = {dev.get('id','')}")
    if probe.get("draft_model"):
        # Without this the engine tries to build an inline MTP head and the
        # model fails to load ("failed to create MTP context").
        L.append(f"spec-draft-model     = {probe['draft_model']}")
    L.append("")
    L.append("; Sampling was held FIXED during the sweep: these are the values the")
    L.append("; numbers above were measured with, not values that were tuned.")
    L.append("temp    = 1.0")
    L.append("top-p   = 0.95")
    L.append("top-k   = 20")
    L.append("min-p   = 0.0")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def cmd_report(a) -> int:
    probe_path = os.path.join(a.out, "probe.json")
    probe = json.load(open(probe_path)) if os.path.exists(probe_path) else {}
    facts = probe.get("facts", {})
    ceil = probe.get("ceilings", {})
    rows = _load_rows(a.out)
    if not rows:
        print("no results found", file=sys.stderr)
        return 1

    # spec rows are the ones that report an acceptance figure
    spec = [r for r in rows if _num(r.get("acc_len"))]
    plain = [r for r in rows if not _num(r.get("acc_len"))]

    def agg(rs, key):
        vals = [_num(r.get(key)) for r in rs]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    # Raw llama-bench rows carry the bandwidth measurement. Compute it up front
    # because it is also the best bandwidth to refit the prediction with: the
    # theoretical peak ignores protocol overhead and lands well above what a
    # matvec-shaped workload actually achieves.
    raw = [r for r in plain if r.get("name", "").startswith("raw_")]

    def _depth(r):
        m = re.search(r"_d(\d+)$", r.get("name", ""))
        return int(m.group(1)) if m else 0

    tg_rows = [r for r in raw if "tg" in r.get("name", "")]
    # llama-bench "-d" appends the depth, so tg rows can come from several
    # contexts; keep depth 0 as the headline and the deepest as the serving
    # figure rather than averaging them together
    tg = agg([r for r in tg_rows if _depth(r) == 0], "tps") or None
    max_depth = max((_depth(r) for r in tg_rows), default=0)
    tg_deep = (agg([r for r in tg_rows if _depth(r) == max_depth], "tps")
               if max_depth else None)
    pp_rows = [r for r in raw if "pp" in r.get("name", "")]
    pp = (agg([r for r in pp_rows if _depth(r) == 0], "prompt_tps")
          or agg([r for r in pp_rows if _depth(r) == 0], "tps")) or None
    pp_deep = (agg([r for r in pp_rows if _depth(r) == max_depth], "prompt_tps")
               or agg([r for r in pp_rows if _depth(r) == max_depth], "tps")) \
        if max_depth else None
    achieved_bw = None
    if tg and facts.get("active_bytes"):
        achieved_bw = facts["active_bytes"] * tg / 1e9
    bw_fit = achieved_bw or probe.get("bandwidth", {}).get("theoretical_gbs")
    theo = probe.get("bandwidth", {}).get("theoretical_gbs")

    # The runs measure at whatever context the prompt filled, not at the --ctx
    # sizing value. Feeding --ctx into the cost model overstates the KV term and
    # inflates the predicted ms/pass (50% too high on a small model where KV is
    # a large share of per-token traffic).
    ptoks = [int(r["prompt_tok"]) for r in spec
             if r.get("prompt_tok", "").isdigit()]
    meas_ctx = int(sum(ptoks) / len(ptoks)) if ptoks else a.ctx

    print("\n" + "=" * 74)
    print("llmfit report")
    print("=" * 74)

    if facts:
        print(f"\nmodel : {facts.get('arch')} {facts.get('name','')} "
              f"{facts.get('size_label','')}  ({facts.get('file_bytes',0)/2**30:.2f} GiB, "
              f"{facts.get('n_params',0)/1e9:.2f} B params)")
        print(f"device: {probe.get('device',{}).get('id')} "
              f"{probe.get('device',{}).get('name')}")

    if facts:
        structure_report(facts, probe, ceil.get("ctx", a.ctx),
                         ceil.get("cache_type", "q4_0"))

    if raw:
        print("\n-- raw benchmark (llama-bench) --")
        if max_depth:
            print(f"  (measured at depth 0 and {max_depth}; both fall as context grows)")
        else:
            print("  (prefill at pp512, decode at depth 0 -- the top of the range;")
            print("   rerun with BENCH_DEPTHS=0,8192 to see the serving regime)")
        if pp:
            print(f"  prefill : {pp:.2f} t/s")
        if pp_deep and pp_deep != pp:
            print(f"  prefill at depth {max_depth}: {pp_deep:.2f} t/s"
                  f"   ({100*(pp_deep/pp-1):+.0f}% vs depth 0)")
        if tg:
            print(f"  decode  : {tg:.2f} t/s")
        if tg_deep and tg_deep != tg:
            print(f"  decode at depth {max_depth}: {tg_deep:.2f} t/s"
                  f"   ({100*(tg_deep/tg-1):+.0f}% vs depth 0)")
        if achieved_bw:
            print(f"  achieved effective bandwidth : {achieved_bw:.1f} GB/s")
            if theo and achieved_bw <= theo:
                print(f"  theoretical bandwidth        : {theo:.1f} GB/s  "
                      f"-> {100*achieved_bw/theo:.0f}% of peak")
            elif theo:
                print(f"  theoretical bandwidth        : {theo:.1f} GB/s  "
                      f"-> achieved is HIGHER than the nominal peak")
                print("     Either the nominal figure (DIMMs x 64-bit x MT/s) is a")
                print("     lower bound for this memory, or the model reads fewer")
                print("     bytes per token than its total size -- selective")
                print("     activation designs (Gemma's E4B) do exactly that.")
        ab = facts.get("active_bytes")
        if tg and theo and ab:
            # llama-bench tg runs at depth 0, so compare like with like: the
            # ctx-sized ceiling below answers a different question and must not
            # be used as the denominator here (it reported >100% efficiency on
            # a small model where KV is a large share of per-token traffic).
            c0 = theo * 1e9 / ab
            print(f"  decode ceiling at depth 0    : {c0:.2f} t/s")
            if tg <= c0:
                print(f"  decode efficiency            : {100*tg/c0:.0f}% of the "
                      f"depth-0 bandwidth ceiling")
            else:
                print(f"  decode efficiency            : above the nominal ceiling"
                      f" (see note above)")
        if ceil.get("decode_ceiling_tps"):
            print(f"  ceiling at ctx={ceil.get('ctx')}         : "
                  f"{ceil['decode_ceiling_tps']:.2f} t/s   (KV read per token"
                  f" at full ctx: {ceil.get('kv_memory',0)/2**20:.0f} MiB)")

    if not spec:
        print("\n  no speculative-decoding results in this run - the sweep did "
              "not complete,\n  so there is nothing to recommend.")
        if a.ini and probe:
            p = write_ini(a.ini, probe, [], "", 0.0, None, [], a.hf, a.out)
            print(f"  wrote an untuned preset anyway: {p}")
        print()
        return 0

    # Rows from a different prompt are not comparable with the main sweep, so
    # they are ranked separately instead of competing for "best".
    marker = f"prompt={a.workload}"
    primary = [r for r in spec if marker in r.get("config", "")]
    other = [r for r in spec if marker not in r.get("config", "")]
    if not primary:
        primary, other = spec, []

    # Group repeats of one configuration. Run-to-run spread on this class of
    # device is a few percent, which is wider than the gaps between the top
    # configurations, so a single lucky round must not decide the winner.
    groups = {}
    for r in primary:
        key = r.get("config", "").replace(marker, "").strip() or r.get("name", "")
        groups.setdefault(key, []).append(r)

    ranked = []
    for key, rs in groups.items():
        tps = [v for v in (_num(r.get("tps")) for r in rs) if v is not None]
        if tps:
            ranked.append((key, sum(tps) / len(tps), len(tps), rs))
    ranked.sort(key=lambda t: -t[1])

    print(f"\n-- speculative decoding, workload '{a.workload}' --")
    print(f"  {'config':<28} {'rounds':>6} {'decode t/s':>10} {'acc_len':>8} {'ms/pass':>8}")
    for key, mean, n, rs in ranked:
        al = [v for v in (_num(r.get("acc_len")) for r in rs) if v is not None]
        mp = [v for v in (_num(r.get("ms_per_pass")) for r in rs) if v is not None]
        print(f"  {key[:28]:<28} {n:>6} {mean:>10.2f} "
              f"{(f'{sum(al)/len(al):.2f}' if al else '-'):>8} "
              f"{(f'{sum(mp)/len(mp):.0f}' if mp else '-'):>8}")

    rounds = max(t[2] for t in ranked) if ranked else 1
    solid = [t for t in ranked if t[2] == rounds]
    best_key, best_mean, _bn, best_rs = max(solid, key=lambda t: t[1])
    print(f"\n  BEST of {rounds} round(s): {best_key} -> {best_mean:.2f} t/s")

    for key, mean, n, _rs in ranked:
        if n < rounds and mean > best_mean * 1.05:
            print(f"  NOT CONFIRMED: {key} posted {mean:.2f} t/s in {n} round(s) "
                  f"({100*(mean/best_mean-1):+.0f}%); re-run with --rounds 2")

    unspec = agg([r for r in plain if "nospec" in r.get("name", "")], "tps")
    if unspec:
        print(f"  no-speculation control   : {unspec:.2f} t/s "
              f"-> {best_mean/unspec:.2f}x")

    # Refit the prediction from the measurement with the largest n_max (most
    # positions constrain the per-position rate) and the achieved bandwidth,
    # since the theoretical peak overstates what a matvec-shaped workload does.
    fit_src, meas = None, {}
    for key, mean, _n, rs in ranked:
        # only the plain n-max sweep: a b/ub or ngram variant measures a
        # different configuration and would bias the curve
        m = re.fullmatch(r"n-max=(\d+) p-min=[\d.]+", key)
        if not m:
            continue
        nmax = int(m.group(1))
        al = [v for v in (_num(r.get("acc_len")) for r in rs) if v is not None]
        if not al:
            continue
        meas[nmax] = mean
        if fit_src is None or nmax > fit_src[0]:
            fit_src = (nmax, sum(al) / len(al))
    a_fit = fit_accept(*fit_src) if fit_src else None

    # Cost model is fitted to the MEASUREMENTS rather than to assumed traffic.
    # Assuming what a drafted token reads gets the step cost wrong by an order of
    # magnitude once the draft is a separate small model rather than the target's
    # own head, and a single line through a cliffless assumption misses the
    # regime change at higher n.
    cost_pts = []
    for key, _mean, _n, rs in ranked:
        m = re.fullmatch(r"n-max=(\d+) p-min=[\d.]+", key)
        if not m:
            continue
        mp = [v for v in (_num(r.get("ms_per_pass")) for r in rs) if v is not None]
        if mp:
            cost_pts.append((int(m.group(1)), sum(mp) / len(mp)))
    fit = fit_pass_cost(cost_pts) if len(cost_pts) >= 2 else None

    if a_fit and fit:
        t_base_ms, t_draft_ms, cliff = fit
        print(f"\n  -- fit from measurements: per-position acceptance {a_fit:.3f} "
              f"(from n_max={fit_src[0]}, mean len {fit_src[1]}) --")
        print(f"     pass cost = {t_base_ms:.1f} ms + n x {t_draft_ms:.1f} ms/step"
              + (f"   (cliff detected at n_max={cliff}, excluded from the fit)"
                 if cliff else ""))
        print(f"  {'n_max':>5} {'exp.len':>8} {'ms/pass':>9} {'predicted':>10} {'measured':>9}")
        for n in sorted(meas) + [max(meas) + 2]:
            exp_len = sum(a_fit ** i for i in range(n + 1))
            cost_ms = t_base_ms + n * t_draft_ms
            m = meas.get(n)
            print(f"  {n:>5} {exp_len:>8.2f} {cost_ms:>9.1f} "
                  f"{exp_len / (cost_ms / 1000):>10.2f} {(f'{m:.2f}' if m else '-'):>9}")
        opt = max(meas, key=lambda k: meas[k]) if meas else None
        if cliff:
            print(f"     n_max={cliff} is off this line: per-step cost jumps, so the"
                  f"\n     linear fit does not apply there and is not extrapolated.")

    if other:
        print(f"\n-- workload sensitivity (prompts other than '{a.workload}') --")
        print(f"  {'config':<44} {'decode t/s':>10} {'acc_len':>8}")
        for r in sorted(other, key=lambda r: -(_num(r["tps"]) or 0)):
            print(f"  {r.get('config','')[:44]:<44} {_fmt(_num(r['tps']),2):>10} "
                  f"{_fmt(_num(r['acc_len']),2):>8}")
        print("  (acceptance length is workload dependent: repetitive or echoing")
        print("   output accepts more drafted tokens than varied prose or code.)")
        lens = [v for v in (_num(r.get("acc_len")) for r in spec) if v is not None]
        if lens:
            print(f"  measured mean len spans {min(lens):.2f}-{max(lens):.2f} across the"
                  f" prompts in this run,")
            print("  so any single figure is workload specific. Point --prompt at your")
            print("  own workload to measure the real one; a higher acceptance favours a")
            print(f"  larger --spec-draft-n-max than the {len(lens) and ''}value tuned here.")

    # ready-to-run command, from the winner of the primary workload
    if facts and ranked:
        dev = probe.get("device", {}).get("id", "Vulkan0")
        m = re.search(r"n-max=(\d+)", best_key)
        nmax = m.group(1) if m else "4"
        spec_type = "draft-mtp,ngram-mod" if "ngram" in best_key else "draft-mtp"
        fa = "off" if "fa=off" in best_key else "on"
        kv = "f16" if "kv=f16" in best_key else ceil.get("cache_type", "q4_0")
        bm = re.search(r"b=(\d+) ub=(\d+)", best_key)
        b, ub = (bm.group(1), bm.group(2)) if bm else ("1024", "512")
        print(f"\n-- recommended ({best_mean:.2f} t/s, workload '{a.workload}') --")
        print(f"  ./llama-server -hf {a.hf or '<repo>:<quant>'} \\")
        print(f"      -dev {dev} -lv 3 --metrics --port 8420 \\")
        print(f"      --spec-type {spec_type} --spec-draft-n-max {nmax} "
              f"--spec-draft-p-min 0.0 \\")
        print(f"      -b {b} -ub {ub} -c {ceil.get('ctx', a.ctx)} -fa {fa} "
              f"-ctk {kv} -ctv {kv} \\")
        print(f"      --parallel 1 --host 127.0.0.1")
    if a.ini and probe:
        p = write_ini(a.ini, probe, ranked, best_key, best_mean, unspec, other,
                      a.hf, a.out)
        print("\n-- llama.cpp model preset written --")
        print(f"  {p}")
        print(f"  use it with: llama-server --models-preset {p}")
        if a.hf:
            print("  (the section name matches the cached model, so the preset is")
            print("   picked up automatically when that model is requested by name)")
        else:
            print("  (the section names a local file and sets model = explicitly,")
            print("   so no cached entry is needed)")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="llmfit_lib.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--binary", required=True)
    p.add_argument("--bench", default="")
    p.add_argument("--hf", default="")
    p.add_argument("--model", default="")
    p.add_argument("--draft-model", default="",
                   help="draft GGUF for speculation, or 'none' to disable")
    p.add_argument("--dev", default="")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--cache-type", default="q4_0")
    p.add_argument("--mem-bus-bits", type=int, default=0)
    p.add_argument("--mem-mt-s", type=int, default=0)
    p.add_argument("--peak-tflops", type=float, default=0.0)
    p.add_argument("--assumed-accept", type=float, default=0.75)
    p.add_argument("--out", default="")
    p.add_argument("--download", action="store_true")
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("parse")
    s.add_argument("--log", required=True)
    s.add_argument("--power", default="")
    s.add_argument("--name", required=True)
    s.add_argument("--config", default="")
    s.add_argument("--summary", required=True)
    s.set_defaults(func=cmd_parse)

    rn = sub.add_parser("replay-run")
    rn.add_argument("--turns", required=True)
    rn.add_argument("--port", type=int, required=True)
    rn.add_argument("--gen", type=int, default=128)
    rn.add_argument("--limit", type=int, default=0)
    rn.add_argument("--seed", type=int, default=42)
    rn.set_defaults(func=cmd_replay_run)

    rr = sub.add_parser("replay-report")
    rr.add_argument("--log", required=True)
    rr.add_argument("--max-turns", type=int, default=0)
    rr.set_defaults(func=cmd_replay_report)

    rp = sub.add_parser("replay-extract")
    rp.add_argument("--session", required=True,
                    help="OMP session JSONL under ~/.omp/agent/sessions/")
    rp.add_argument("--out", required=True, help="output directory")
    rp.add_argument("--max-chars", type=int, default=400000,
                    help="cap each cumulative prompt (keeps the recent tail)")
    rp.add_argument("--include-thinking", action="store_true",
                    help="include assistant thinking blocks in the prompt")
    rp.add_argument("--show-usage", action="store_true",
                    help="also report the recorded token counts and ttft")
    rp.set_defaults(func=cmd_replay_extract)

    b = sub.add_parser("raw")
    b.add_argument("--bench-log", required=True)
    b.add_argument("--summary", required=True)
    b.add_argument("--prefix", default="raw_")
    b.add_argument("--config", default="")
    b.set_defaults(func=cmd_raw)

    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--hf", default="")
    r.add_argument("--ctx", type=int, default=131072)
    r.add_argument("--workload", default="varied",
                   help="prompt marker to rank as the primary workload")
    r.add_argument("--ini", default="",
                   help="write a llama.cpp model-preset INI to this path")
    r.set_defaults(func=cmd_report)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
