#!/usr/bin/env python3
"""
llm_hardware_check.py (v2.3.0) - checks which LLMs from several providers fit your hardware.

Providers (--provider):
    hf          Hugging Face (GGUF repos)           [default]
    ollama      Ollama library (registry.ollama.ai)
    modelscope  ModelScope (GGUF repos)
    all         all three

The script
  1. detects VRAM (NVIDIA via nvidia-smi, Apple Silicon as unified memory) and RAM,
  2. searches the provider for models (or checks a specific model),
  3. reads the real file sizes per quantization,
  4. estimates the total memory need (weights + KV cache + overhead),
  5. classifies every variant: GPU full / MoE offload / Partial offload / CPU only / Too large.

No third-party dependencies: runs on Python 3.9+ (standard library only).
Optional: environment variable HF_TOKEN for private/gated Hugging Face repos.

Examples:
    python llm_hardware_check.py --all                       (most important models, all providers)
    python llm_hardware_check.py --search qwen --limit 10
    python llm_hardware_check.py --provider all --search llama --only-fit
    python llm_hardware_check.py --provider ollama --repo qwen2.5 --context 16384
    python llm_hardware_check.py --provider ollama --repo qwen2.5:14b
    python llm_hardware_check.py --repo hf:bartowski/Qwen2.5-14B-Instruct-GGUF --repo ollama:llama3.1
    python llm_hardware_check.py --search mistral --vram 12 --ram 32
    python llm_hardware_check.py --mac "m4 max 64" --provider all --search qwen
    python llm_hardware_check.py --mac "m4 max 64" --provider all --search qwen --html report.html
    python llm_hardware_check.py --vram 24 --ram 64 --bandwidth 1008 --search llama   (e.g. RTX 4090)

More models: several search terms and a higher limit (per provider and term):
    python llm_hardware_check.py --all-macs --provider all --limit 15 --search qwen --search llama --search gemma

Compare all Macs (M1 to M5, common memory sizes). The result is an interactive HTML report:
choose hardware, context and efficiency, filter by provider, size, architecture (dense/MoE),
quantization and minimum quantization bit depth, sort, export as CSV:
    python llm_hardware_check.py --all-macs --provider all --search qwen --limit 5
    python llm_hardware_check.py --all-macs --mac "m4 max 64" --search qwen   (your own Mac is marked)

Tokens/s estimate: chip bandwidth (from a table with --mac, otherwise --bandwidth)
times efficiency (--efficiency, default 0.6) divided by the weights read actively per token.
A self-contained HTML report is always written at the end (disable with --no-html).

Cache: provider responses are stored for 24 hours under ~/.cache/llm-hardware-check; a second
run (e.g. with another Mac or filter) then takes only seconds. --refresh reloads everything,
--no-cache disables the cache, --cache-ttl HOURS changes the lifetime.

Metadata: for Hugging Face and Ollama the start of the GGUF file (256 KiB) is read via a range request.
This yields layers, KV heads and expert count: the KV cache is then computed instead of estimated,
and MoE models are recognized even without "A3B" in the name. --no-meta skips this.

Ollama context: Ollama loads models with a default context that depends on GPU memory (under 24 GiB 4k,
from 24 GiB 32k, from 48 GiB 256k tokens, at most the model's context). By default the script uses
this for Ollama models; --ollama-ctx same uses --context instead.

Version 2.2 additions:
    --calibrate ollama:qwen2.5:14b=28.5   store your measured tokens/s; the efficiency is then calibrated per machine
    --tflops N / --prompt-tokens N        prompt processing (prefill) and time to first token
    --live                                use the currently free memory instead of RAM minus a fixed reserve
    --diff                                new / updated models since the last run of the same query
    --commands                            print a download/start command for the best variant
  The HTML report can compare two hardware configurations side by side and copy commands per variant.
  Sliding-window models (Gemma 2/3/3n, OLMo 3, gpt-oss, Cohere 2) are calculated with their smaller KV cache.

--repo can take a prefix: hf:, ollama:, ms:
Without a prefix the provider chosen with --provider applies (for "all": hf).

Note: all values are estimates. For MoE models (e.g. Mixtral, Qwen MoE) the memory need
and speed estimates depend on the architecture and runtime; neither is guaranteed. Ollama and ModelScope
have no officially documented search API; if the search fails, use --repo.
"""

from __future__ import annotations

__version__ = "2.7.0"

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import html as html_lib

GB = 1024 ** 3
OVERHEAD_GB = 1.0          # runtime overhead (CUDA context, buffers)
# baseline OS usage including typical background programs (GB, assumption; change with --reserve)
OS_RESERVE_GB = {"macos": 6.0, "windows": 8.0, "linux": 3.0}
GPU_RESERVE_GB = {"macos": 0.0, "windows": 1.0, "linux": 0.5}   # desktop/driver with a dedicated GPU
OS_LABEL = {"macos": "macOS", "windows": "Windows", "linux": "Linux"}
SETTINGS = {"os": None, "reserve": None, "mac_limit": "reserve", "ollama_ctx": "std", "live": False, "ram_bw": None}   # set in main()
Q4_GB_PER_B = 0.57         # fallback: GB per billion parameters at about Q4

QUANT_RE = re.compile(r"(I?Q\d(?:_[A-Z0-9]+)*|NVFP4|MXFP4|BF16|F16|F32)", re.IGNORECASE)
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}")
PARAMS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
UA = {"User-Agent": f"llm-hardware-check/{__version__}"}


# ---------------------------------------------------------------- Hardware --
@dataclass
class Hardware:
    gpu_name: str
    vram_gb: float
    ram_gb: float
    unified: bool = False
    gpu_cap_gb: float = 0.0    # usable GPU memory (unified: share of total memory)
    ram_cap_gb: float = 0.0    # usable RAM for offloading (RAM minus operating system)
    note: str = ""             # where the limits come from


def host_os():
    return {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")


def os_reserve(os_key):
    return SETTINGS["reserve"] if SETTINGS["reserve"] is not None else OS_RESERVE_GB[os_key]


def unified_cap(ram_gb, reserve=None):
    """Apple Silicon: usable memory. Default: RAM minus operating system (requires a raised
    GPU limit, 'sudo sysctl iogpu.wired_limit_mb=...'); alternatively the macOS default limit."""
    if SETTINGS["mac_limit"] == "default":
        return ram_gb * (2 / 3 if ram_gb <= 36 else 3 / 4)
    return max(0.0, ram_gb - (os_reserve("macos") if reserve is None else reserve))


def unified_note(ram_gb):
    if SETTINGS["mac_limit"] == "default":
        return "macOS default limit (2/3 up to 36 GB, otherwise 3/4 of memory)"
    return f"{ram_gb:.0f} GB minus {os_reserve('macos'):.0f} GB for macOS (raise the GPU limit via sysctl)"


def detect_ram_gb() -> float:
    if platform.system() == "Windows":
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMSTATUS()
        stat.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / GB
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (ValueError, OSError, AttributeError):
        return 0.0


def detect_free_ram_gb():
    """Currently available memory (including reclaimable caches) in GiB, or None."""
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024 / GB
        elif platform.system() == "Darwin":
            out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = sum(int(m.group(1)) for k in ("free", "inactive", "speculative", "purgeable")
                        for m in [re.search(rf"Pages {k}:\s+(\d+)", out)] if m)
            return pages * page / GB
        elif platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong * 5)]
            st = MS()
            st.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ap / GB
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    return None


def detect_nvidia_free_gb():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
        return sum(float(x) for x in out.split()) / 1024
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def detect_nvidia():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        gpus = []
        for line in out.strip().splitlines():
            name, mem = [p.strip() for p in line.rsplit(",", 1)]
            gpus.append((name, float(mem) / 1024))
        if gpus:
            name = gpus[0][0] if len(gpus) == 1 else f"{len(gpus)}x {gpus[0][0]}"
            return name, sum(g[1] for g in gpus)
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return None


def detect_hardware(vram_override, ram_override, force_unified=False) -> Hardware:
    ram = ram_override if ram_override is not None else detect_ram_gb()
    if force_unified or (platform.system() == "Darwin" and platform.machine() == "arm64"):
        cap = vram_override if vram_override is not None else unified_cap(ram)
        note = "set manually" if vram_override is not None else unified_note(ram)
        if SETTINGS["live"] and vram_override is None:
            free = detect_free_ram_gb()
            if free is not None:
                cap = min(cap, free)
                note += f"; live: {free:.1f} GB currently free"
        return Hardware("Apple Silicon (Unified Memory)", cap, ram, unified=True, gpu_cap_gb=cap, note=note)
    key = SETTINGS["os"] or host_os()
    reserve = os_reserve(key)
    ram_cap = max(0.0, ram - reserve)
    nv = detect_nvidia()
    if vram_override is not None:
        vram, name = vram_override, (nv[0] if nv else "Custom GPU")
    elif nv:
        vram, name = nv[1], nv[0]
    else:
        vram, name = 0.0, "no GPU detected"
    gpu_cap = max(0.0, vram - GPU_RESERVE_GB[key]) if vram > 0 else 0.0
    note = f"VRAM minus {GPU_RESERVE_GB[key]:.1f} GB (desktop/driver); RAM minus {reserve:.0f} GB for {OS_LABEL[key]}"
    if SETTINGS["live"]:
        free = detect_free_ram_gb()
        if free is not None:
            ram_cap = min(ram_cap, free)
            note += f"; live: {free:.1f} GB RAM free"
        fv = detect_nvidia_free_gb() if (vram_override is None and nv) else None
        if fv is not None:
            gpu_cap = min(gpu_cap, max(0.0, fv - 0.1))
            note += f", {fv:.1f} GB VRAM free"
    return Hardware(name, vram, ram, gpu_cap_gb=gpu_cap, ram_cap_gb=ram_cap, note=note)


CHIP_BANDWIDTH = {  # memory bandwidth in GB/s (manufacturer figures, rounded)
    "m1": 68, "m1 pro": 200, "m1 max": 400, "m1 ultra": 800,
    "m2": 100, "m2 pro": 200, "m2 max": 400, "m2 ultra": 800,
    "m3": 100, "m3 pro": 150, "m3 max": 400, "m3 ultra": 819,
    "m4": 120, "m4 pro": 273, "m4 max": 546,
    "m5": 153.6, "m5 pro": 307, "m5 max": 614,
}
# Max chips with only 36 GB have the smaller GPU and therefore less bandwidth
SMALL_MAX_BANDWIDTH = {"m3 max": 300, "m4 max": 410, "m5 max": 460}

# GPU compute (FP16/FP32 TFLOPS, rounded, approximate) for the prompt-processing (prefill) estimate
CHIP_TFLOPS = {
    "m1": 2.6, "m1 pro": 5.2, "m1 max": 10.4, "m1 ultra": 21.0,
    "m2": 3.6, "m2 pro": 6.8, "m2 max": 13.6, "m2 ultra": 27.2,
    "m3": 4.1, "m3 pro": 7.4, "m3 max": 16.4, "m3 ultra": 32.8,
    "m4": 4.3, "m4 pro": 9.2, "m4 max": 18.4,
    "m5": 5.0, "m5 pro": 10.0, "m5 max": 20.0,
}
NVIDIA_TFLOPS = (   # (regex on the GPU name, FP16 tensor TFLOPS dense, approximate)
    (r"5090", 210.0), (r"4090", 165.0), (r"4080", 97.0), (r"3090", 71.0), (r"3080", 60.0),
    (r"H100", 756.0), (r"H200", 989.0), (r"A100", 312.0), (r"L40S", 362.0), (r"A6000", 155.0), (r"5080", 113.0), (r"6000 ?Ada", 366.0),
)
# presets for the HTML report: (name, VRAM GB, bandwidth GB/s, TFLOPS or None = table lookup); several cards: bandwidth of one card
GPU_PRESETS = (
    ("RTX 3090", 24, 936, None), ("RTX 4090", 24, 1008, None), ("RTX 5080", 16, 960, None), ("RTX 5090", 32, 1792, None),
    ("2x RTX 3090", 48, 936, 71.0), ("2x RTX 4090", 48, 1008, 165.0), ("2x RTX 5090", 64, 1792, 210.0),
    ("4x RTX 4090", 96, 1008, 165.0), ("RTX 6000 Ada", 48, 960, None), ("RTX PRO 6000 Blackwell", 96, 1792, None),
    ("A100 80GB", 80, 2039, None), ("H100 80GB", 80, 3350, None), ("H200 141GB", 141, 4800, None),
    ("4x H100 80GB", 320, 3350, 756.0), ("8x H100 80GB", 640, 3350, 756.0),
)
NVIDIA_BANDWIDTH = (   # (regex on the GPU name, memory bandwidth GB/s, manufacturer figures, rounded; first match wins)
    (r"5090", 1792), (r"5080", 960), (r"5070 ?Ti", 896), (r"5070", 672),
    (r"4090", 1008), (r"4080 ?SUPER", 736), (r"4080", 717), (r"4070 ?Ti ?SUPER", 672), (r"4070 ?Ti", 504), (r"4070", 504),
    (r"3090 ?Ti", 1008), (r"3090", 936), (r"3080 ?Ti", 912), (r"3080", 760), (r"3070", 448), (r"3060", 360),
    (r"RTX PRO 6000", 1792), (r"6000 ?Ada", 960), (r"A6000", 768), (r"L40S?", 864),
    (r"H200", 4800), (r"H100.*(NVL|SXM)|H100 80GB HBM3", 3350), (r"H100", 2000), (r"B200", 8000),
    (r"A100.*40", 1555), (r"A100", 2039),
)
NVIDIA_LAPTOP_BANDWIDTH = (   # laptop GPUs (GB/s, typical configuration; manufacturers vary memory clocks)
    (r"5090", 896), (r"5080", 896), (r"5070 ?Ti", 672), (r"5070", 384), (r"5060", 384),
    (r"4090", 576), (r"4080", 432), (r"4070", 256), (r"4060", 256), (r"4050", 192),
    (r"3080 ?Ti", 512), (r"3080", 448), (r"3070 ?Ti", 448), (r"3070", 448), (r"3060", 336),
)
PREFILL_EFF = 0.3          # share of the peak compute reached when processing the prompt (assumption)

# typical memory configurations per chip (GB); approximate, Apple does not offer every combination in every device
MAC_CONFIGS = [
    ("M1", (8, 16)), ("M1 Pro", (16, 32)), ("M1 Max", (32, 64)), ("M1 Ultra", (64, 128)),
    ("M2", (8, 16, 24)), ("M2 Pro", (16, 32)), ("M2 Max", (32, 64, 96)), ("M2 Ultra", (64, 128, 192)),
    ("M3", (8, 16, 24)), ("M3 Pro", (18, 36)), ("M3 Max", (36, 48, 64, 96, 128)), ("M3 Ultra", (96, 256, 512)),
    ("M4", (16, 24, 32)), ("M4 Pro", (24, 48, 64)), ("M4 Max", (36, 48, 64, 128)),
    ("M5", (12, 16, 24, 32)), ("M5 Pro", (24, 48, 64)), ("M5 Max", (36, 48, 64, 128)),
]


def detect_mac_chip():
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return out.replace("Apple", "").strip() or None


def chip_bandwidth(chip, ram_gb):
    """Table value; Max chips with only 36 GB have the smaller GPU (e.g. M4 Max: 410 instead of 546 GB/s)."""
    if not chip:
        return None
    key = re.sub(r"\s+", " ", chip.lower()).strip()
    if key in SMALL_MAX_BANDWIDTH and ram_gb and ram_gb < 48:
        return SMALL_MAX_BANDWIDTH[key]
    return CHIP_BANDWIDTH.get(key)


def chip_tflops(chip, ram_gb):
    if not chip:
        return None
    key = re.sub(r"\s+", " ", chip.lower()).strip()
    tf = CHIP_TFLOPS.get(key)
    if tf and key in SMALL_MAX_BANDWIDTH and ram_gb and ram_gb < 48:   # smaller GPU
        tf *= SMALL_MAX_BANDWIDTH[key] / CHIP_BANDWIDTH[key]
    return tf


def is_laptop_gpu(name):
    return bool(re.search(r"laptop|max-q|mobile", name or "", re.IGNORECASE))


def gpu_bandwidth(name):
    """Table lookup by GPU name (several cards: bandwidth of one card, layers are split across them).
    Laptop GPUs only use the laptop table: the desktop values would be far too high."""
    for pattern, bw in (NVIDIA_LAPTOP_BANDWIDTH if is_laptop_gpu(name) else NVIDIA_BANDWIDTH):
        if re.search(pattern, name or "", re.IGNORECASE):
            return bw
    return None


def gpu_tflops(name):
    if is_laptop_gpu(name):   # power-limited, no reliable figure: no first-token estimate instead of a desktop value
        return None
    for pattern, tf in NVIDIA_TFLOPS:
        if re.search(pattern, name or "", re.IGNORECASE):
            return tf
    return None


# ---------------------------------------------------------- Data model ------
@dataclass
class Variant:
    label: str
    size_gb: float
    params_b: float | None = None
    meta: dict | None = None   # Ollama only: GGUF metadata per tag (layers, KV, experts, context)
    ident: str | None = None   # Ollama only: blob digest, equal digests = alias tags of the same file


@dataclass
class Candidate:
    provider: str
    name: str
    variants: list = field(default_factory=list)
    downloads: int | None = None
    error: str | None = None
    usage: list = field(default_factory=list)   # use cases, e.g. ["Chat/Text", "Code"]
    updated: str | None = None                   # last update (ISO date), if known
    meta: dict = field(default_factory=dict)    # from the GGUF header: arch, layers, kv_bpt, experts, experts_used


def params_from_name(text: str) -> float | None:
    matches = PARAMS_RE.findall(text)
    return max(float(m) for m in matches) if matches else None


SKIPPED = []   # variants dropped because of missing sizes or incomplete shard sets: (repo, file)


def group_gguf_files(files, repo=""):
    """files: iterable of (path, size_bytes). Merges the shards of a model.

    Several different files with the same quantization (e.g. finetune variants in one repo)
    stay separate entries and are not summed up."""
    sets: dict[tuple, float] = {}
    shards = {}
    bad = set()
    for path, size in files:
        if not path.lower().endswith(".gguf") or "mmproj" in path.lower() or "projector" in path.lower():
            continue
        if re.search(r"(^|/)(mtp|draft)/", path, re.IGNORECASE):  # auxiliary files (draft models), not a variant of their own
            continue
        folder, base = os.path.split(path)
        key = (folder, SHARD_RE.sub("", base))
        shard = re.search(r"-(\d{5})-of-(\d{5})", base)
        if shard:
            index, total = map(int, shard.groups())
            shards.setdefault(key, []).append((index, total))
        if not size or size <= 0:
            bad.add(key)   # file without a usable size: the variant would be wrong, so it is skipped
            continue
        sets[key] = sets.get(key, 0.0) + size / GB
    for key, parts in shards.items():
        totals = {total for _, total in parts}
        total = parts[0][1]
        indices = [index for index, _ in parts]
        if (len(totals) != 1 or total < 1 or len(indices) != total
                or set(indices) != set(range(1, total + 1))):
            bad.add(key)
    for key in bad:
        sets.pop(key, None)
        SKIPPED.append((repo, key[1]))
    entries = []
    for (folder, base), size in sets.items():
        m = QUANT_RE.search(base) or QUANT_RE.search(folder)
        entries.append((folder, base, m.group(1).upper() if m else "?", size))
    counts: dict[str, int] = {}
    for _, _, quant, _ in entries:
        counts[quant] = counts.get(quant, 0) + 1
    groups: dict[str, float] = {}
    for folder, base, quant, size in entries:
        if quant == "?":   # no quantization in the name: show the file name instead of "?"
            label = base[:-5] if base.lower().endswith(".gguf") else base
            if folder:
                label = f"{folder}/{label}"
        elif counts[quant] == 1:
            # only show the folder if it says more than the quantization itself
            label = f"{folder}/{quant}" if folder and quant.lower() not in folder.lower() else quant
        else:
            label = base[:-5] if base.lower().endswith(".gguf") else base
            if folder:
                label = f"{folder}/{label}"
        groups[label] = groups.get(label, 0.0) + size
    return groups


# ---------------------------------------------------------------- HTTP ------
class _Cache:
    def __init__(self):
        self.dir = None
        self.ttl = 24 * 3600
        self.refresh = False
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()


CACHE = _Cache()


def cache_setup(enabled, ttl_hours, refresh):
    """Cache provider responses on disk (default: 24 hours)."""
    CACHE.dir = None
    if not enabled:
        return
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "llm-hardware-check")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return
    CACHE.dir, CACHE.ttl, CACHE.refresh = path, ttl_hours * 3600, refresh


def cached(key, producer):
    """Store the result of producer() (JSON-serializable) under key; errors are never stored."""
    if not CACHE.dir:
        return producer()
    fn = os.path.join(CACHE.dir, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")
    if not CACHE.refresh:
        try:
            with open(fn, encoding="utf-8") as fh:
                entry = json.load(fh)
            if time.time() - entry["t"] <= CACHE.ttl:
                with CACHE.lock:
                    CACHE.hits += 1
                return entry["v"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    value = producer()
    with CACHE.lock:
        CACHE.misses += 1
    try:
        tmp = f"{fn}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"t": time.time(), "v": value}, fh)
        os.replace(tmp, fn)
    except (OSError, TypeError, ValueError):
        pass
    return value


# ------------------------------------------------------------------ TLS -----
SSL_CTX = None
SSL_NOTE = ""


def _verify_paths_ok():
    p = ssl.get_default_verify_paths()
    if p.cafile and os.path.isfile(p.cafile):
        return True
    return bool(p.capath and os.path.isdir(p.capath) and os.listdir(p.capath))


def ssl_setup(ca_bundle=None):
    """TLS context for all requests. Certificate checks always stay on.
    --ca-bundle / SSL_CERT_FILE: own CA file (e.g. a company or intercepting proxy CA).
    Python from python.org on macOS ships without CA certificates until "Install Certificates.command" was run;
    then the certifi package is used if installed, otherwise the macOS system root certificates (via `security`)."""
    global SSL_CTX, SSL_NOTE
    if ca_bundle:
        SSL_CTX = ssl.create_default_context(cafile=ca_bundle)
        SSL_NOTE = f"CA certificates from {ca_bundle}"
        return
    SSL_CTX = ssl.create_default_context()
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR") or _verify_paths_ok():
        return
    try:
        import certifi   # optional, not a requirement
        SSL_CTX.load_verify_locations(certifi.where())
        SSL_NOTE = "CA certificates from certifi (Python has no own CA store)"
        return
    except (ImportError, OSError, ssl.SSLError):
        pass
    if platform.system() == "Darwin" and shutil.which("security"):
        try:
            pem = subprocess.check_output(
                ["security", "find-certificate", "-a", "-p",
                 "/System/Library/Keychains/SystemRootCertificates.keychain"], text=True, timeout=20)
            if "BEGIN CERTIFICATE" in pem:
                SSL_CTX.load_verify_locations(cadata=pem)
                SSL_NOTE = "CA certificates from the macOS system keychain (Python has no own CA store)"
        except (subprocess.SubprocessError, OSError, ssl.SSLError):
            pass


def ssl_hint(errors):
    """One clear hint instead of a certificate error per model."""
    if not any("CERTIFICATE_VERIFY_FAILED" in e or "SSLCertVerification" in e or "SSL:" in e for e in errors):
        return None
    msg = ("TLS certificate check failed. Typical causes: Python from python.org on macOS without CA certificates "
           "(run \"/Applications/Python 3.x/Install Certificates.command\" or `pip install certifi`), an intercepting "
           "proxy such as Burp/ZAP in HTTPS_PROXY (unset it or pass its CA with --ca-bundle FILE), or outdated "
           "ca-certificates (Linux).")
    return msg


def _urlopen(req, timeout):
    return urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX)


def http_get(url, headers=None, timeout=20):
    def go():
        req = urllib.request.Request(url, headers={**UA, **(headers or {})})
        with _urlopen(req, timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    if headers and "Authorization" in headers:   # never store requests with a token on disk
        return go()
    return cached("GET " + url, go)


def http_json(url, body=None, headers=None, timeout=20, method=None):
    def go():
        data = None
        hdrs = {**UA, **(headers or {})}
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        with _urlopen(req, timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    if headers and "Authorization" in headers:
        return go()
    key = f"{method or ('POST' if body is not None else 'GET')} {url} {json.dumps(body, sort_keys=True)}"
    return cached(key, go)


# ----------------------------------------------------- GGUF metadata --------
GGUF_HEAD_BYTES = 262144   # the architecture keys usually come before the tokenizer, a few KiB are normally enough
_GGUF_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
               6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_GGUF_WANT = {"block_count", "embedding_length", "context_length", "expert_count", "expert_used_count",
              "attention.head_count", "attention.head_count_kv", "attention.key_length",
              "attention.value_length", "attention.sliding_window", "expert_feed_forward_length",
              "leading_dense_block_count"}


def parse_gguf_header(buf):
    """Reads the architecture metadata from the start of a GGUF file.

    Usually the architecture keys come before the (huge) tokenizer and a small slice is enough;
    in some files they come after it, in which case the tokenizer is skipped.
    With truncated data the values read so far are returned."""
    out = {}
    if buf[:4] != b"GGUF":
        return out
    try:
        pos = 8
        _, n_kv = struct.unpack_from("<QQ", buf, pos)
        pos += 16

        def read(t, pos):
            if t in _GGUF_FIXED:
                fmt, size = _GGUF_FIXED[t]
                return struct.unpack_from(fmt, buf, pos)[0], pos + size
            if t == 8:
                (n,) = struct.unpack_from("<Q", buf, pos)
                pos += 8
                if pos + n > len(buf):
                    raise struct.error("truncated")
                return buf[pos:pos + n].decode("utf-8", "replace"), pos + n
            if t == 9:
                et, n = struct.unpack_from("<IQ", buf, pos)
                pos += 12
                items = []
                for _ in range(n):
                    v, pos = read(et, pos)
                    items.append(v)
                return items, pos
            raise ValueError(t)

        def skip(t, pos):
            if t in _GGUF_FIXED:
                return pos + _GGUF_FIXED[t][1]
            if t == 9:
                et, n = struct.unpack_from("<IQ", buf, pos)
                pos += 12
                if et in _GGUF_FIXED:
                    return pos + n * _GGUF_FIXED[et][1]
                if et == 8:   # string list (tokenizer): read only the lengths, do not decode
                    for _ in range(n):
                        (ln,) = struct.unpack_from("<Q", buf, pos)
                        pos += 8 + ln
                    if pos > len(buf):
                        raise struct.error("truncated")
                    return pos
            return read(t, pos)[1]

        arch = None
        for _ in range(n_kv):
            key, pos = read(8, pos)
            (t,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            if key == "general.architecture":
                arch, pos = read(t, pos)
                out["arch"] = arch
            elif arch and key.startswith(arch + ".") and key[len(arch) + 1:] in _GGUF_WANT:
                out[key[len(arch) + 1:]], pos = read(t, pos)
            else:
                pos = skip(t, pos)
        out["_complete"] = pos <= len(buf)
    except (struct.error, ValueError, IndexError):
        pass
    return out


def read_gguf_head(url, headers):
    """Read the start of a GGUF file via a range request. If the architecture keys are missing (large fields
    before them), read again with a larger slice."""
    h = {}
    for size in (GGUF_HEAD_BYTES, 8 * GGUF_HEAD_BYTES, 32 * GGUF_HEAD_BYTES):
        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes=0-{size - 1}"})
        with _urlopen(req, 30) as resp:
            buf = resp.read(size)
        h = parse_gguf_header(buf)
        # A layer count alone is insufficient: KV/expert fields may follow.
        if (h.get("_complete") or len(buf) < size or buf[:4] != b"GGUF"
                or (kv_bytes_per_token(h) is not None
                    and "expert_count" in h and "expert_used_count" in h
                    and "context_length" in h)):
            break
    return h


def kv_bytes_per_token(h):
    """KV cache per token in bytes (f16): layers x KV heads x (key + value length) x 2 bytes."""
    layers = h.get("block_count")
    heads = h.get("attention.head_count")
    kv_heads = h.get("attention.head_count_kv") or heads
    if not layers or not kv_heads:
        return None
    emb = h.get("embedding_length")
    head0 = max(heads) if isinstance(heads, list) else heads
    key_len = h.get("attention.key_length") or (emb // head0 if emb and head0 else None)
    if not key_len:
        return None
    val_len = h.get("attention.value_length") or key_len
    total_heads = sum(kv_heads) if isinstance(kv_heads, list) else kv_heads * layers
    return int(total_heads * (key_len + val_len) * 2)


SWA_ARCH = {   # architecture: (default window, share of layers with global attention); the rest uses the window
    "gemma2": (4096, 1 / 2), "gemma3": (1024, 1 / 6), "gemma3n": (512, 1 / 5),
    "olmo3": (4096, 1 / 4), "gpt-oss": (128, 1 / 2), "cohere2": (4096, 1 / 4),
}


def swa_of(h):
    """[window, global share] for models with sliding-window attention (known architectures only)."""
    arch = h.get("arch")
    if arch in SWA_ARCH:
        window = h.get("attention.sliding_window")
        window = window if isinstance(window, int) and window > 0 else SWA_ARCH[arch][0]
        return [window, round(SWA_ARCH[arch][1], 4)]
    return None


def kv_gb_for(kv_bpt, swa, ctx):
    """KV cache in GiB; with sliding-window attention only the global layers grow with the full context."""
    eff = ctx
    if swa:
        window, glob = swa
        eff = ctx * glob + min(ctx, window) * (1 - glob)
    return kv_bpt * eff / GB


def meta_from_header(h):
    if not h:
        return {}
    meta = {"arch": h.get("arch"), "layers": h.get("block_count"), "ctx_train": h.get("context_length"),
            "kv_bpt": kv_bytes_per_token(h), "experts": h.get("expert_count"),
            "experts_used": h.get("expert_used_count"), "swa": swa_of(h),
            "emb": h.get("embedding_length"), "ffe": h.get("expert_feed_forward_length"),
            "leading_dense": h.get("leading_dense_block_count")}
    return {k: v for k, v in meta.items() if v}


def meta_active_fraction(meta):
    n, k = meta.get("experts"), meta.get("experts_used")
    if n and k and 0 < k < n:
        return min(1.0, k / n + 0.03)   # selected experts plus shared layers
    return 1.0


# ------------------------------------------------------- Provider: HF -------
HF_API = "https://huggingface.co/api/models"


USAGE_ORDER = ["Chat/Text", "Code", "Reasoning", "Vision", "Tools", "Math", "Embedding",
               "Image", "Audio", "Uncensored"]
_PIPELINE_USAGE = (
    ("text-generation", "Chat/Text"), ("text2text", "Chat/Text"), ("conversational", "Chat/Text"),
    ("chat", "Chat/Text"), ("image-text-to-text", "Vision"), ("visual-question", "Vision"),
    ("image-to-text", "Vision"), ("multi-modal", "Vision"), ("multimodal", "Vision"),
    ("text-to-image", "Image"), ("image-to-image", "Image"), ("feature-extraction", "Embedding"),
    ("sentence-similarity", "Embedding"), ("embedding", "Embedding"), ("speech", "Audio"),
    ("audio", "Audio"), ("text-to-video", "Image"),
)
_NAME_USAGE = (
    ("Code", r"coder|codestral|devstral|starcoder|codellama|(^|[-_/.])code([-_.]|$)"),
    ("Reasoning", r"thinking|reason|(^|[-_/.])r1([-_.]|$)|qwq|magistral|[-_.]think"),
    ("Vision", r"(^|[-_/.])(vl|vision|llava|pixtral|moondream|paligemma|minicpm-v|smolvlm|ocr)([-_.]|$)|olmocr"),
    ("Math", r"math"),
    ("Embedding", r"embed|rerank|(^|[-_/.])(bge|e5|gte|minilm)([-_.]|$)"),
    ("Image", r"(^|[-_/.])(image|diffusion|flux|sdxl|sd3)([-_.]|$)"),
    ("Audio", r"whisper|(^|[-_/.])(tts|asr|stt)([-_.]|$)|voxtral|speech|audio"),
    ("Uncensored", r"uncensored|abliterat|heretic|dolphin|no-?refusal"),
    ("Chat/Text", r"instruct|chat|(^|[-_.])it([-_.]|$)|assistant"),
)


def derive_usage(name, pipeline=None, tags=None, caps=None, tasks=None):
    """Use cases of a model: from provider metadata (if available) and name patterns."""
    found = set()
    for p in [pipeline] + list(tasks or []):
        p = str(p or "").lower()
        for key, label in _PIPELINE_USAGE:
            if key in p:
                found.add(label)
    meta = [str(t).lower() for t in list(tags or []) + list(caps or [])]
    for t in meta:
        if t in ("vision", "image-text-to-text"):
            found.add("Vision")
        elif t in ("thinking", "reasoning"):
            found.add("Reasoning")
        elif t in ("tools", "tool-use", "function-calling", "tool-calling"):
            found.add("Tools")
        elif t in ("embedding", "embeddings"):
            found.add("Embedding")
        elif t == "conversational":
            found.add("Chat/Text")
    n = str(name).lower().split("/")[-1]   # not the owner (e.g. "Embedding-GGUF/...")
    for label, pattern in _NAME_USAGE:
        if re.search(pattern, n):
            found.add(label)
    if found & {"Embedding", "Image", "Audio"}:
        found.discard("Chat/Text")  # do not show pure non-text models as chat
    return [u for u in USAGE_ORDER if u in found]


def parse_count(num, unit=""):
    try:
        value = float(num.replace(",", "."))
    except ValueError:
        return None
    return int(value * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}.get(unit.upper(), 1))


def hf_headers():
    token = os.environ.get("HF_TOKEN")  # optional, only for private/gated repos
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_search(query, limit):
    params = {"filter": "gguf", "sort": "downloads", "direction": "-1", "limit": limit}
    if query:
        params["search"] = query
    qs = urllib.parse.urlencode(params)
    data = http_json(f"{HF_API}?{qs}", headers=hf_headers())
    out = []
    for m in data:
        name = m.get("id") or m.get("modelId")
        out.append((name, m.get("downloads"),
                    derive_usage(name, m.get("pipeline_tag"), m.get("tags"))))
    return out


_HF_INFO = {}


def hf_variants(repo_id):
    data = http_json(f"{HF_API}/{repo_id}?blobs=true", headers=hf_headers())
    _HF_INFO[repo_id] = data
    params_b = None
    gguf = data.get("gguf")
    if isinstance(gguf, dict) and gguf.get("total"):
        params_b = gguf["total"] / 1e9
    if params_b is None:
        params_b = params_from_name(repo_id.split("/")[-1])
    pairs = []
    for f in data.get("siblings") or []:
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        pairs.append((f.get("rfilename", ""), size))
    return [Variant(k, s, params_b) for k, s in group_gguf_files(pairs, repo_id).items()]


def hf_meta(repo_id):
    """Architecture metadata from the start of a GGUF file (HTTP range, 256 KiB)."""
    data = _HF_INFO.get(repo_id) or {}
    pick = None
    for f in data.get("siblings") or []:
        name = f.get("rfilename", "")
        low = name.lower()
        if not low.endswith(".gguf") or "mmproj" in low or "projector" in low or "imatrix" in low:
            continue
        if re.search(r"(^|/)(mtp|draft)/", name, re.IGNORECASE):
            continue
        if SHARD_RE.search(name) and "-00001-of-" not in name:
            continue
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        if size < 100 * 1024 * 1024:
            continue
        if pick is None or size < pick[0]:
            pick = (size, name)
    if not pick:
        return {}
    url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(pick[1])}"

    def fetch():
        attempts = [{}] + ([hf_headers()] if hf_headers() else [])
        last = None
        for headers in attempts:
            try:
                return read_gguf_head(url, {**UA, **headers})
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (401, 403):
                    raise
        raise last

    return meta_from_header(fetch() if hf_headers() else cached("GGUFHEAD5 " + url, fetch))


META_FETCHERS = {"hf": hf_meta}


# -------------------------------------------------- Provider: ModelScope ----
MS_HOSTS = ("https://www.modelscope.cn", "https://modelscope.cn")


def ms_search(query, limit):
    # The search API is not officially documented: try several variants.
    last_err = None
    for host in MS_HOSTS:
        for method in ("PUT", "POST"):
            for with_filter in (True, False):
                body = {
                    "PageSize": limit, "PageNumber": 1, "SortBy": "DownloadsCount", "Target": "",
                    "SingleCriterion": [], "Name": query,
                    "Criterion": ([{"category": "libraries", "predicate": "contains",
                                    "values": ["GGUF"], "IsPredicate": True}] if with_filter else []),
                }
                try:
                    data = http_json(f"{host}/api/v1/dolphin/models", body, method=method)
                except Exception as exc:
                    last_err = exc
                    continue
                models = (data.get("Data") or {}).get("Model", {}).get("Models") or []
                out = []
                for m in models:
                    path, name = m.get("Path"), m.get("Name")
                    if path and name and (with_filter or "gguf" in name.lower()):
                        tasks = []
                        for t in m.get("Tasks") or []:
                            tasks.append(t.get("Name") or t.get("name") or "" if isinstance(t, dict) else t)
                        out.append((f"{path}/{name}", m.get("Downloads") or m.get("DownloadsCount"),
                                    derive_usage(name, tasks=tasks)))
                if out:
                    return out[:limit]
    raise RuntimeError(f"ModelScope search returned nothing (last error: {last_err}) - "
                       "use --repo ms:owner/name")


def ms_variants(repo_id):
    data, last_err = None, None
    for host in MS_HOSTS:
        try:
            data = http_json(f"{host}/api/v1/models/{repo_id}/repo/files?Recursive=true")
            break
        except Exception as exc:
            last_err = exc
    if data is None:
        raise last_err
    files = (data.get("Data") or {}).get("Files") or []
    pairs = [(f.get("Path") or f.get("Name") or "", f.get("Size") or 0)
             for f in files if f.get("Type", "blob") == "blob"]
    params_b = params_from_name(repo_id.split("/")[-1])
    return [Variant(k, s, params_b) for k, s in group_gguf_files(pairs, repo_id).items()]


# ----------------------------------------------------- Provider: Ollama -----
OLLAMA_REGISTRY = "https://registry.ollama.ai/v2"
OLLAMA_ACCEPT = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}


def ollama_ns(name):
    return name if "/" in name else f"library/{name}"


_OLLAMA_CAPS = {"vision", "tools", "thinking", "embedding", "cloud"}


def ollama_card_info(chunk):
    """Capabilities and pull count from the HTML of a model card; first via the x-test attributes,
    otherwise via the visible text (lines like "vision", "tools", "3.8M")."""
    caps = re.findall(r'x-test-capability[^>]*>\s*([A-Za-z]+)\s*<', chunk)
    pulls = re.search(r'x-test-pull-count[^>]*>\s*([\d.,]+)\s*([KMBkmb]?)\s*<', chunk)
    downloads = parse_count(pulls.group(1), pulls.group(2)) if pulls else None
    if not caps or downloads is None:
        lines = [ln.strip() for ln in re.sub(r"<[^>]+>", "\n", chunk).split("\n")]
        lines = [html_lib.unescape(ln) for ln in lines if ln]
        if not caps:
            caps = [ln.lower() for ln in lines if ln.lower() in _OLLAMA_CAPS]
        if downloads is None:
            for ln in lines:
                m = re.fullmatch(r"(\d[\d.,]*)\s*([KMB])", ln)
                if m:
                    downloads = parse_count(m.group(1), m.group(2))
                    break
    return caps, downloads


def ollama_search(query, limit):
    html = http_get(f"https://ollama.com/search?q={urllib.parse.quote(query)}")
    hits = []
    for m in re.finditer(r'href="/library/([^"/?#:]+)"', html):
        if m.group(1) not in [h[0] for h in hits]:
            hits.append((m.group(1), m.start()))
    if not hits:
        raise RuntimeError("Ollama search returned nothing - use --repo ollama:modelname")
    out = []
    for i, (name, pos) in enumerate(hits[:limit]):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(html)
        chunk = html[pos:end]   # model card (capabilities, pull count), as far as readable
        caps, downloads = ollama_card_info(chunk)
        out.append((name, downloads, derive_usage(name, caps=caps)))
    return out


def ollama_tags(name):
    ns = ollama_ns(name)
    try:
        tags = http_json(f"{OLLAMA_REGISTRY}/{ns}/tags/list").get("tags") or []
        if tags:
            return tags
    except Exception:
        pass
    html = http_get(f"https://ollama.com/{ns}/tags")
    pattern = re.escape(ns) + r":([A-Za-z0-9_.\-]+)"
    return sorted(set(re.findall(pattern, html)))


def ollama_head(ns, digest):
    """GGUF metadata of an Ollama model: start of the model blob via a range request."""
    url = f"{OLLAMA_REGISTRY}/{ns}/blobs/{digest}"

    def fetch():
        return read_gguf_head(url, dict(UA))

    return meta_from_header(cached("GGUFHEAD5 " + url, fetch))


def ollama_variant(name, tag):
    ns = ollama_ns(name)
    mf = http_json(f"{OLLAMA_REGISTRY}/{ns}/manifests/{tag}", headers=OLLAMA_ACCEPT)
    layers = [layer for layer in mf.get("layers", []) if layer.get("mediaType", "").endswith("image.model")]
    size = sum(layer.get("size", 0) for layer in layers)
    if not size:
        return None
    params_b = params_from_name(tag) or params_from_name(name)
    label = tag if tag != "latest" else "latest (usually Q4)"
    meta = None
    if USE_META and layers and layers[0].get("digest"):
        try:
            meta = ollama_head(ns, layers[0]["digest"]) or None
        except Exception:
            meta = None   # optional: without metadata the rough estimate applies
    return Variant(label, size / GB, params_b, meta, layers[0].get("digest") if layers else None)


def dedupe_aliases(variants):
    """Tags with the same blob (e.g. 7b, 7b-q4_0, latest) are one file: keep only the most descriptive name."""
    best, order = {}, []
    for v in variants:
        if not v.ident:
            order.append(v)
            continue
        cur = best.get(v.ident)
        if cur is None:
            best[v.ident] = v
            order.append(v)
            continue
        def score(x):
            return (not x.label.startswith("latest"), bool(QBITS_RE.search(x.label)) or is_full_precision(x.label), len(x.label))
        if score(v) > score(cur):
            order[order.index(cur)] = v
            best[v.ident] = v
    return order


def ollama_variants(name):
    if ":" in name:  # specific tag, e.g. qwen2.5:14b
        base, tag = name.split(":", 1)
        v = ollama_variant(base, tag)
        return [Variant(f"{base}:{v.label}", v.size_gb, v.params_b, v.meta)] if v else []
    tags = list(dict.fromkeys(ollama_tags(name)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda t: safe(ollama_variant, name, t), tags))
    return dedupe_aliases([v for v in results if v])


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


# ---------------------------------------------------------- Provider map ----
PROVIDERS = {
    "hf": ("Hugging Face", hf_search, hf_variants),
    "ollama": ("Ollama", ollama_search, ollama_variants),
    "modelscope": ("ModelScope", ms_search, ms_variants),
}
PREFIXES = {"hf": "hf", "ollama": "ollama", "ms": "modelscope", "modelscope": "modelscope"}


# ------------------------------------------------------------ Evaluation -----
def estimate_kv_gb(params_b, size_gb, context: int) -> float:
    """Rough estimate: about 0.5 GB per billion parameters at 32k context (GQA models)."""
    if params_b is None:
        params_b = size_gb / Q4_GB_PER_B
    return 0.5 * params_b * context / 32768


ICONS = {"GPU full": "[++]", "MoE offload": "[+M]", "Partial offload": "[+ ]", "CPU only": "[~ ]", "Too large": "[--]"}
ORDER = {"GPU full": 0, "MoE offload": 1, "Partial offload": 2, "CPU only": 3, "Too large": 4}
RAM_BW_DEFAULT = 60.0      # GB/s of system RAM actually reached (dual-channel DDR5 desktop); --ram-bandwidth
MOE_EXPERT_SHARE = 0.9     # share of the file taken by the expert tensors when neither header nor name tell more


def moe_split(cand, v, frac):
    """MoE models: (expert share of the file, share of the experts read per token, number of MoE layers).
    The expert share is computed from the GGUF header (experts x 3 x embedding x expert FFN x MoE layers);
    without it the share is derived from the active share in the name (30B-A3B -> 95 %, 80B-A3B -> 98 %).
    None for dense models."""
    m = v.meta or cand.meta or {}
    n, k = m.get("experts"), m.get("experts_used")
    known = bool(n and k and 0 < k < n)
    if frac >= 1.0 and not known:
        return None
    layers = m.get("layers")
    moe_layers = layers - (m.get("leading_dense") or 0) if layers else None
    share = None
    if known and m.get("emb") and m.get("ffe") and moe_layers and v.params_b:
        share = n * 3 * m["emb"] * m["ffe"] * moe_layers / (v.params_b * 1e9)
        if not 0.3 < share < 0.995:
            share = None
    if share is None:   # without header data: the smaller the active share, the larger the expert share
        share = min(0.98, max(0.85, 1 - frac / 2)) if frac < 1.0 else MOE_EXPERT_SHARE
    # share of the expert weights read per token; from the name: active share minus the always-read dense part
    ratio = k / n if known else max(0.005, (frac - (1 - share)) / share)
    return (round(share, 4), round(ratio, 4), moe_layers)


def place(size_gb, kv_gb, frac, split, hw, bandwidth, efficiency, moe_ok=True):
    """Where a variant runs and how fast. Returns (need, verdict, tokens/s, share of the experts on the CPU).
    MoE offload (llama.cpp --n-cpu-moe): attention, shared parts and KV cache stay in VRAM, as many expert layers
    as fit as well, the rest of the experts goes to RAM; per token only the active experts are read from RAM.
    Partial offload: whole layers are split between VRAM and RAM. Both are limited by the RAM bandwidth."""
    need = size_gb + kv_gb + OVERHEAD_GB
    gpu_cap, ram_cap = hw.gpu_cap_gb, hw.ram_cap_gb
    ram_bw = SETTINGS.get("ram_bw") or RAM_BW_DEFAULT
    gbs = lambda x: x * 1e9 / GB          # GB/s -> GiB/s
    if gpu_cap > 0 and need <= gpu_cap:
        return need, "GPU full", estimate_tps(size_gb, frac, bandwidth, efficiency), 0.0
    if hw.unified:
        return need, "Too large", None, 0.0
    if split and moe_ok and gpu_cap > 0:
        share, ratio, _ = split
        exp = size_gb * share
        dense = size_gb - exp
        on_cpu = need - gpu_cap
        if not split[2]:     # layer count unknown: the command can only move all experts (--cpu-moe)
            on_cpu = exp
        if dense + kv_gb + OVERHEAD_GB <= gpu_cap and on_cpu <= exp and on_cpu <= ram_cap:
            c = on_cpu / exp
            tps = None
            if bandwidth:
                t = (dense + exp * ratio * (1 - c)) / gbs(bandwidth) + exp * ratio * c / gbs(ram_bw)
                tps = efficiency / max(t, 1e-6)
            return need, "MoE offload", tps, c
    if gpu_cap > 0 and need <= gpu_cap + ram_cap:
        g = min(1.0, max(0.0, (gpu_cap - kv_gb - OVERHEAD_GB) / size_gb))
        active = max(size_gb * frac, 0.1)
        tps = efficiency / (active * g / gbs(bandwidth) + active * (1 - g) / gbs(ram_bw)) if bandwidth else None
        return need, "Partial offload", tps, 0.0
    if need <= ram_cap:
        return need, "CPU only", efficiency * gbs(ram_bw) / max(size_gb * frac, 0.1), 0.0
    return need, "Too large", None, 0.0


MOE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b[-_ ]?a(\d+(?:\.\d+)?)\s*b", re.IGNORECASE)
NX_RE = re.compile(r"(?<![\d.])(\d{1,2})x(\d+(?:\.\d+)?)b", re.IGNORECASE)
FULL_PREC_RE = re.compile(r"(?<![a-z0-9])(f32|f16|bf16|fp16|fp32)(?![a-z0-9])", re.IGNORECASE)


def active_fraction(text):
    """MoE models are recognized by name: 30B-A3B (3 of 30 billion active) or Mixtral-style 8x22B."""
    m = MOE_RE.search(text)
    if m:
        total, active = float(m.group(1)), float(m.group(2))
        if 0 < active < total:
            return active / total
    m = NX_RE.search(text)
    if m and int(m.group(1)) >= 4:  # Mixtral-style: 2 active experts plus shared layers
        return min(1.0, 2 / int(m.group(1)) + 0.03)
    return 1.0


def is_full_precision(label):
    return bool(FULL_PREC_RE.search(label))


def estimate_tps(size_gb, frac, bandwidth, efficiency):
    """Token generation is bandwidth-limited: the active weights are read once per token."""
    if not bandwidth:
        return None
    return efficiency * (bandwidth * 1e9 / GB) / max(size_gb * frac, 0.1)


def estimate_prefill(size_gb, params_b, frac, tflops):
    """Prompt processing is compute-bound: about 2 FLOPs per active parameter and token."""
    if not tflops:
        return None
    params = (params_b if params_b else size_gb / Q4_GB_PER_B) * 1e9
    return PREFILL_EFF * tflops * 1e12 / (2 * params * frac)


@dataclass
class Row:
    label: str
    size_gb: float
    need_gb: float
    verdict: str
    tps: float | None = None
    moe: bool = False
    params_b: float | None = None
    ctx: int = 0              # context actually used
    pf: float | None = None   # prompt processing (prefill) in tokens/s
    ncpu: str | None = None   # MoE offload: value for llama.cpp --n-cpu-moe ("all" = --cpu-moe)


def ollama_default_ctx(mem_gb):
    """Ollama's default context depends on GPU memory (official documentation):
    under 24 GiB 4k, from 24 GiB 32k, from 48 GiB 256k tokens."""
    return 4096 if mem_gb < 24 else (32768 if mem_gb < 48 else 262144)


def effective_context(cand, v, hw, context):
    """Context the model is loaded with: for Ollama the default by memory size,
    at most the model's context (unknown: 32k)."""
    if cand.provider != "ollama" or SETTINGS["ollama_ctx"] != "std":
        return context
    mem = hw.gpu_cap_gb if hw.unified else hw.vram_gb
    train = (v.meta or cand.meta or {}).get("ctx_train")
    return min(ollama_default_ctx(mem), train or 32768)


def prepare_variants(cand, quant_filter):
    """Filters out non-models and determines the active parameter share per variant (MoE)."""
    # alias tags (e.g. "35b" = "35b-a3b") have the same file size: transfer MoE detection to them
    size_moe = {}
    for v in cand.variants:
        f = active_fraction(f"{cand.name} {v.label}")
        if f < 1.0:
            size_moe[round(v.size_gb, 1)] = f
    out = []
    for v in cand.variants:
        if v.size_gb < 0.15:  # tiny files are not models (tokenizer, imatrix etc.)
            continue
        if v.params_b and v.size_gb < 0.15 * v.params_b:  # smaller than any real quantization: auxiliary file
            continue
        if quant_filter and not any(q in v.label.upper() for q in quant_filter):
            continue
        frac = size_moe.get(round(v.size_gb, 1), active_fraction(f"{cand.name} {v.label}"))
        if frac >= 1.0:
            frac = meta_active_fraction(v.meta or cand.meta)   # expert count from the GGUF header, if the name does not reveal it
        out.append((v, frac))
    return out


def evaluate(cand: Candidate, hw, context, quant_filter, bandwidth=None, efficiency=0.6, tflops=None):
    rows = []
    for v, frac in prepare_variants(cand, quant_filter):
        ctx = effective_context(cand, v, hw, context)
        m = v.meta or cand.meta or {}
        kv_bpt = m.get("kv_bpt")
        kv_gb = kv_gb_for(kv_bpt, m.get("swa"), ctx) if kv_bpt else estimate_kv_gb(v.params_b, v.size_gb, ctx)
        split = moe_split(cand, v, frac)
        need, verdict, tps, c = place(v.size_gb, kv_gb, frac, split, hw, bandwidth, efficiency,
                                      moe_ok=cand.provider != "ollama")
        pf = estimate_prefill(v.size_gb, v.params_b, frac, tflops) if verdict == "GPU full" else None
        ncpu = None
        if verdict == "MoE offload":
            ncpu = str(min(split[2], math.ceil(c * split[2] + 1e-9))) if split[2] else "all"
            if split[2] and int(ncpu) >= split[2]:
                ncpu = "all"
        rows.append(Row(v.label, v.size_gb, need, verdict, tps, frac < 1.0 or split is not None, v.params_b, ctx, pf, ncpu))
    rows.sort(key=lambda r: (ORDER[r.verdict], -r.size_gb))
    return rows


QBITS_RE = re.compile(r"(?:^|[^a-z0-9])i?q(\d)", re.IGNORECASE)


def quant_bits(label, size_gb, params_b):
    """Bits per weight: first from the variant name (Q4_K_M -> 4), otherwise from file size and parameters."""
    m = QBITS_RE.search(label)
    if m:
        return float(m.group(1))
    if re.search(r"f32|fp32", label, re.IGNORECASE):
        return 32.0
    if is_full_precision(label):
        return 16.0
    return size_gb * 1.0737 * 8 / params_b if params_b else None


QCLASS_ORDER = ["FP", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q1", "?"]
QCLASS_LABEL = {"FP": "BF16/F16/F32", "Q8": "Q8", "Q6": "Q6", "Q5": "Q5", "Q4": "Q4", "Q3": "Q3",
                "Q2": "Q2", "Q1": "Q1", "?": "not specified"}


def quant_class(label):
    """Quantization class of a variant: Q8, Q6, ..., Q1 (also IQ variants), FP for BF16/F16/F32."""
    if is_full_precision(label):
        return "FP"
    m = QBITS_RE.search(label)
    return f"Q{m.group(1)}" if m else "?"


def best_moe(rows):
    """Largest variant that runs with MoE offload (if nothing fits completely)."""
    pool = [r for r in rows if r.verdict == "MoE offload" and not is_full_precision(r.label)]
    return max(pool, key=lambda r: r.size_gb) if pool else None


def best_fit(rows):
    """Largest variant that fits completely; full precision (F16/BF16/F32) only if nothing else fits."""
    fits = [r for r in rows if r.verdict == "GPU full"]
    pool = [r for r in fits if not is_full_precision(r.label)] or fits
    return max(pool, key=lambda r: r.size_gb) if pool else None


USE_META = True


def updated_epoch(cand):
    if not cand.updated:
        return None
    try:
        return int(datetime.fromisoformat(str(cand.updated).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


UPD_STEPS = (("30", "last 30 days"), ("90", "last 90 days"), ("365", "last year"))


def usage_of(cand):
    return cand.usage or derive_usage(cand.name)


def dedupe_mirrors(cands):
    """Remove repeated provider/repository entries, preserving distinct repositories.

    Equal names and rounded file sizes do not establish file identity.
    Cross-repository mirrors must remain visible without verified content hashes.
    """
    best = {}
    for cand in cands:
        key = (cand.provider, cand.name)
        previous = best.get(key)
        if previous is None or (previous.error and not cand.error):
            best[key] = cand
    return list(best.values()), len(cands) - len(best)


def load_candidate(provider, name, downloads, usage=None):
    _, _, variants_fn = PROVIDERS[provider]
    cand = Candidate(provider, name, downloads=downloads, usage=usage or derive_usage(name))
    try:
        cand.variants = variants_fn(name)
    except Exception as exc:
        cand.error = f"{type(exc).__name__}: {exc}"
        return cand
    if provider == "hf":
        cand.updated = (_HF_INFO.get(name) or {}).get("lastModified")
    fetch_meta = META_FETCHERS.get(provider)
    if fetch_meta and cand.variants and USE_META:
        try:
            cand.meta = fetch_meta(name) or {}
        except Exception:
            cand.meta = {}   # metadata is optional: on errors the rough estimate applies
    fill_sibling_meta(cand)
    return cand


def fill_sibling_meta(cand):
    """Variants without their own metadata (header fetch failed) take over that of siblings with the same
    parameter count or name prefix - otherwise e.g. a bf16 tag with a roughly estimated KV cache
    would be compared against the exactly computed Q8/Q4 tags."""
    have = [w for w in cand.variants if (w.meta or {}).get("kv_bpt")]
    if not have:
        return
    for v in cand.variants:
        if (v.meta or {}).get("kv_bpt"):
            continue
        key = v.label.split("-")[0].split(":")[-1]
        for w in have:
            same_params = v.params_b and w.params_b and abs(w.params_b - v.params_b) <= 0.05 * v.params_b
            if same_params or (not v.params_b and w.label.split("-")[0].split(":")[-1] == key):
                v.meta = w.meta
                break


# ------------------------------------------- State: calibration, snapshots --
def state_dir():
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    path = os.path.join(base, "llm-hardware-check")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path


def _state_read(name, default):
    d = state_dir()
    if d:
        try:
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return default


def _state_write(name, value):
    d = state_dir()
    if not d:
        return False
    try:
        tmp = os.path.join(d, f"{name}.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
        os.replace(tmp, os.path.join(d, name))
        return True
    except (OSError, TypeError, ValueError):
        return False


def hw_key(hw, chip):
    """Key of a machine for stored calibration values."""
    if hw.unified:
        return f"{(chip or 'apple silicon').lower()} {hw.ram_gb:g}"
    return (hw.gpu_name or "gpu").lower()


def stored_efficiency(key):
    entry = _state_read("calibration.json", {}).get(key)
    return (entry.get("eff"), len(entry.get("samples", []))) if entry and entry.get("eff") else (None, 0)


def calibrate(specs, key, bandwidth, default_provider):
    """--calibrate SPEC=TOKENS_PER_SECOND with SPEC = [hf:|ollama:|ms:]name[@variant] (Ollama: name:tag).
    Computes the efficiency that reproduces your measured speed and stores the median per machine."""
    import statistics
    if not bandwidth:
        print("Calibration needs the memory bandwidth (--mac or --bandwidth).")
        return
    data = _state_read("calibration.json", {})
    entry = data.setdefault(key, {"samples": []})
    for spec in specs:
        target, sep, tps_txt = spec.rpartition("=")
        try:
            tps = float(tps_txt)
            if not sep or tps <= 0:
                raise ValueError
        except ValueError:
            print(f"Ignored (expected SPEC=TOKENS_PER_SECOND): {spec}")
            continue
        provider, rest = default_provider, target
        prefix, psep, tail = target.partition(":")
        if psep and prefix.lower() in PREFIXES and tail:
            provider, rest = PREFIXES[prefix.lower()], tail
        label = None
        if "@" in rest:
            rest, label = rest.split("@", 1)
        elif provider == "ollama" and ":" in rest:
            pass
        cand = load_candidate(provider, rest, None)
        if cand.error or not cand.variants:
            print(f"Not found: {target} ({cand.error or 'no variants'})")
            continue
        rows = prepare_variants(cand, None)
        pick = None
        for v, frac in rows:
            lab = v.label.lower()
            if label is None or lab == label.lower() or lab.endswith(":" + label.lower()) or label.lower() in lab:
                pick = (v, frac)
                break
        if pick is None or (label is None and len(rows) > 1 and not (provider == "ollama" and ":" in rest)):
            print(f"Ambiguous: {target} - append @VARIANT (variants: {', '.join(v.label for v, _ in rows[:8])} ...)")
            continue
        v, frac = pick
        eff = tps * max(v.size_gb * frac, 0.1) / (bandwidth * 1e9 / GB)
        if not 0.02 <= eff <= 1.5:
            print(f"Implausible value for {target} ({eff:.2f}) - check the variant / speed.")
            continue
        entry["samples"].append({"spec": target, "tps": tps, "eff": round(eff, 3)})
        print(f"{v.label}: {tps:g} tokens/s measured -> efficiency {eff:.0%}")
    entry["samples"] = entry["samples"][-20:]
    if entry["samples"]:
        entry["eff"] = round(min(1.0, statistics.median(x["eff"] for x in entry["samples"])), 3)
        _state_write("calibration.json", data)
        print(f"Stored efficiency for '{key}': {entry['eff']:.0%} (median of {len(entry['samples'])} measurements). "
              f"It is used automatically unless you pass --efficiency.")


def snapshot_sig(selected, searches, repos):
    return hashlib.sha1(json.dumps([sorted(selected), sorted(searches or []), sorted(repos or [])]).encode()).hexdigest()[:12]


def snapshot_entry(c):
    return {"u": c.updated, "v": sorted(v.label for v in c.variants), "d": c.downloads}


CHANGES = {}    # (provider, name) -> "new" / "upd", filled with --diff


def diff_snapshot(cands, sig, show):
    """Compares with the previous run of the same query and saves the current state."""
    name = f"snapshot-{sig}.json"
    prev = _state_read(name, None)
    cur = {f"{c.provider}|{c.name}": snapshot_entry(c) for c in cands if not c.error}
    if prev and show:
        old = prev.get("models", {})
        for key, e in cur.items():
            o = old.get(key)
            prov, nm = key.split("|", 1)
            if o is None:
                CHANGES[(prov, nm)] = "new"
            elif (e["u"] and e["u"] != o.get("u")) or e["v"] != o.get("v"):
                CHANGES[(prov, nm)] = "upd"
        gone = [k for k in old if k not in cur]
        when = datetime.fromtimestamp(prev.get("t", 0)).strftime("%Y-%m-%d %H:%M")
        n_new = sum(1 for x in CHANGES.values() if x == "new")
        n_upd = sum(1 for x in CHANGES.values() if x == "upd")
        print(f"\nChanges since the last run ({when}): {n_new} new, {n_upd} updated, {len(gone)} no longer listed")
        for (prov, nm), kind in list(CHANGES.items())[:25]:
            print(f"  [{kind}] {PROVIDERS[prov][0]}: {nm}")
        if len(CHANGES) > 25:
            print(f"  ... and {len(CHANGES) - 25} more (filter 'Since last run' in the HTML report)")
        msg = f"Changes since {when}: {n_new} new, {n_upd} updated, {len(gone)} no longer listed."
    elif show:
        msg = "--diff: first run for this query, nothing to compare yet. The state was saved."
        print("\n" + msg)
    else:
        msg = None
    _state_write(name, {"t": time.time(), "models": cur})
    return msg


def make_command(provider, name, label, ctx, ncpu=None):
    """Shell command to download/start a variant (with --n-cpu-moe for MoE offload)."""
    if provider == "ollama":
        lab = (label or "").split(" ")[0]
        if ":" in lab:
            ref = lab
        elif lab in ("", "latest"):
            ref = name
        else:
            ref = f"{name}:{lab}"
        return f"ollama pull {ref}\nOLLAMA_CONTEXT_LENGTH={ctx} ollama run {ref}"
    plain = bool(label) and label != "?" and "/" not in label and re.fullmatch(r"[A-Za-z0-9_.-]+", label) is not None
    moe = "" if not ncpu else (" --cpu-moe" if ncpu == "all" else f" --n-cpu-moe {ncpu}")
    if provider == "hf":
        if plain and QUANT_RE.fullmatch(label):
            return f"llama-server -hf {name}:{label} -c {ctx} -ngl 99{moe}"
        return f"hf download {name} --include \"*{label}*\"\n# then: llama-server -m <file>.gguf -c {ctx} -ngl 99{moe}"
    return f"modelscope download --model {name} --include \"*{label}*\"\n# then: llama-server -m <file>.gguf -c {ctx} -ngl 99{moe}"


# ---------------------------------------------------------------- Output ----
def print_top(results, show_tps, only_fit=False, commands=False, ctx=8192):
    """Compact default view: per model the largest variant that fits completely."""
    if only_fit:
        results = [(c, [r for r in rows if r.verdict in ("GPU full", "MoE offload", "Partial offload")]) for c, rows in results]
        results = [(c, rows) for c, rows in results if rows]
    # pure non-text models (embedding, image, audio) are left out of the overview; they stay in the HTML report
    text = [(c, rows) for c, rows in results if not (usage_of(c) and set(usage_of(c)) <= {"Embedding", "Image", "Audio"})]
    hidden = len(results) - len(text)
    results = text
    counts = {}
    for c, _ in results:
        counts[short_name(c)] = counts.get(short_name(c), 0) + 1

    def label_of(c):
        n = short_name(c)
        if counts[n] > 1 and "/" in c.name:   # same name in several repos: show the owner
            n = c.name.split("/")[0] + "/" + n
        return n + {"new": " (new)", "upd": " (upd)"}.get(CHANGES.get((c.provider, c.name)), "")

    picked = []
    for cand, rows in sorted(results, key=lambda cr: -(cr[0].downloads or 0)):
        if not rows:
            continue
        best = best_fit(rows)
        picked.append((cand, best, best or best_moe(rows) or min(rows, key=lambda r: r.size_gb)))
    wn = min(52, max([len(label_of(c)) for c, _, _ in picked] + [5]) + 2)
    wv = min(30, max([len(r.label) for _, _, r in picked] + [7]) + 2)

    def ctx_txt(r):
        c = r.ctx or ctx
        return f"{c // 1024}k" if c >= 1024 else str(c)

    print(f"\n{'Model':<{wn}}{'Source':<8}{'Variant':<{wv}}{'File':>8}{'Need':>8}{'Ctx':>6}{'t/s':>6}  Verdict")
    fits = 0
    for cand, best, r in picked:
        src = {"hf": "HF", "ollama": "Ollama", "modelscope": "MS"}.get(cand.provider, cand.provider)
        name = label_of(cand)[:wn - 1]
        head = f"{name:<{wn}}{src:<8}{r.label[:wv - 1]:<{wv}}{r.size_gb:>7.1f}G{r.need_gb:>7.1f}G{ctx_txt(r):>6}"
        if best:
            fits += 1
            tps = f"{best.tps:.0f}" if (show_tps and best.tps) else "-"
            print(f"{head}{tps:>6}  GPU full")
            if commands:
                for line in make_command(cand.provider, cand.name, best.label, best.ctx or ctx).split("\n"):
                    print(f"    $ {line}")
        elif r.verdict == "MoE offload":
            tps = f"{r.tps:.0f}" if (show_tps and r.tps) else "-"
            print(f"{head}{tps:>6}  MoE offload ({'all' if r.ncpu == 'all' else r.ncpu} expert layers in RAM)")
            if commands:
                for line in make_command(cand.provider, cand.name, r.label, r.ctx or ctx, r.ncpu).split("\n"):
                    print(f"    $ {line}")
        else:
            print(f"{head}{'-':>6}  {r.verdict} (smallest variant)")
    print(f"\n{fits} of {len(picked)} models run with at least one variant completely on the GPU / in usable memory.")
    n_moe = sum(1 for _, b, r in picked if not b and r.verdict == "MoE offload")
    if n_moe:
        print(f"{n_moe} more MoE models run with expert offloading (llama.cpp --n-cpu-moe; "
              f"t/s with {SETTINGS.get('ram_bw') or RAM_BW_DEFAULT:g} GB/s RAM bandwidth, --ram-bandwidth).")
    if hidden:
        print(f"{hidden} non-text models (embedding, image, audio) are hidden here; the HTML report lists them.")
    print("Ctx = context the memory need is calculated for (Ollama: its default by memory; "
          "use --ollama-ctx same to compare all providers at --context).")


def print_candidate(cand, rows, only_fit, show_tps=False, show_pf=None, commands=False, ctx=8192):
    rows = [r for r in rows if not only_fit or r.verdict in ("GPU full", "MoE offload", "Partial offload")]
    if not rows:
        return False
    label = PROVIDERS[cand.provider][0]
    d = f"  |  {cand.downloads:,} downloads" if cand.downloads else ""
    u = f"  |  {', '.join(usage_of(cand))}" if usage_of(cand) else ""
    print(f"\n[{label}] {cand.name}{d}{u}")
    if cand.meta:
        bits = [str(cand.meta.get("arch") or "")]
        if cand.meta.get("layers"):
            bits.append(f"{cand.meta['layers']} layers")
        if cand.meta.get("kv_bpt"):
            bits.append(f"KV {cand.meta['kv_bpt'] / 1024:.0f} KiB/token")
        if cand.meta.get("experts"):
            bits.append(f"{cand.meta['experts']} experts, {cand.meta.get('experts_used', '?')} active")
        print("  Metadata: " + ", ".join(b for b in bits if b))
    og, co = origin_of(cand.name)
    if co:
        print(f"  Developer: {og}, {co}")
    if cand.provider == "ollama" and SETTINGS["ollama_ctx"] == "std" and rows:
        lo, hi = min(r.ctx for r in rows), max(r.ctx for r in rows)
        print(f"  Context (Ollama default by memory): {lo if lo == hi else f'{lo} to {hi}'} tokens")
    head = f"  {'Variant':<28}{'File':>9}{'Need':>9}"
    if show_tps:
        head += f"{'Tokens/s':>10}"
    if show_pf:
        head += f"{'Prompt t/s':>12}{f'TTFT {show_pf // 1024}k':>10}"
    print(head + "  Verdict")
    for r in rows:
        line = f"  {r.label:<28}{r.size_gb:>7.1f}GB{r.need_gb:>7.1f}GB"
        if show_tps:
            t = "-" if r.tps is None else f"~{r.tps:.0f}{'*' if r.moe else ''}"
            line += f"{t:>10}"
        if show_pf:
            line += f"{(f'~{r.pf:.0f}' if r.pf else '-'):>12}{(f'{show_pf / r.pf:.1f} s' if r.pf else '-'):>10}"
        extra = ""
        if r.verdict == "MoE offload":
            extra = " (all expert layers in RAM)" if r.ncpu == "all" else f" ({r.ncpu} expert layers in RAM)"
        print(f"{line}  {ICONS[r.verdict]} {r.verdict}{extra}")
    if commands:
        best = best_fit(rows) or best_moe(rows)
        if best:
            for line in make_command(cand.provider, cand.name, best.label, best.ctx or ctx, best.ncpu).split("\n"):
                print(f"    $ {line}")
    return True


# ------------------------------------------------------------ HTML report ---
HTML_CSS = """
:root{color-scheme:light dark;--bg:#f5f6f8;--fg:#1c2127;--muted:#5b6672;--card:#fff;--line:#dde1e6;--accent:#2457d6;
--v0:#0f6b37;--v0b:#ddf2e5;--v1:#7a5200;--v1b:#fcefcf;--v2:#5b4a8a;--v2b:#e9e4f6;--v3:#9a1f1f;--v3b:#fbe2e2;--v4:#0b5f6b;--v4b:#d8f0f3}
@media (prefers-color-scheme:dark){:root{--bg:#121518;--fg:#e7eaee;--muted:#98a2ad;--card:#1b1f24;--line:#2c3239;--accent:#86a9ff;
--v0:#7fe0a6;--v0b:#16342a;--v1:#f1c76b;--v1b:#3a2f12;--v2:#c3b2f0;--v2b:#2c2646;--v3:#ff9f9f;--v3b:#3e1d1d;--v4:#7fd6e0;--v4b:#14343a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.6rem;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}
a{color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:1.2rem;font-weight:600}
.card .s{color:var(--muted);font-size:.85rem}
h2{font-size:1.1rem;margin:0 0 8px}
h2 small{color:var(--muted);font-weight:400;font-size:.85rem;margin-left:6px}
details.fold{margin:0 0 14px}
details.fold>summary{color:var(--fg);font-size:1rem;cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;margin:0 0 8px;user-select:none}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary::before{content:"\\25B8";color:var(--muted);transition:transform .15s}
details.fold[open]>summary::before{transform:rotate(90deg)}
details.fold>summary h2{margin:0;display:inline}
.tablebox{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{width:100%;border-collapse:collapse}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-size:.78rem;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
tr[hidden],section[hidden]{display:none}
section.model{margin:0 0 24px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:24px 0 16px}
.controls input[type=search]{flex:1 1 240px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
.controls label{display:flex;gap:6px;align-items:center;font-size:.9rem}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}
.badge.v0{background:var(--v0b);color:var(--v0)}.badge.v1{background:var(--v1b);color:var(--v1)}
.badge.v2{background:var(--v2b);color:var(--v2)}.badge.v3{background:var(--v3b);color:var(--v3)}.badge.v4{background:var(--v4b);color:var(--v4)}
.bar{display:inline-block;width:70px;height:8px;background:var(--line);border-radius:4px;margin-right:8px;vertical-align:middle;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.tag{font-size:.7rem;border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:6px;color:var(--muted)}
.use{margin-top:2px}.use .tag{margin:0 4px 0 0}
.controls select{padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
.notes{color:var(--muted);font-size:.9rem}
footer{color:var(--muted);font-size:.85rem;margin-top:32px;border-top:1px solid var(--line);padding-top:12px}
"""

# --------------------------------------------------------- Mac comparison ---
MAC_CSS = """
th[data-sk],th.sortable{cursor:pointer;user-select:none}
th[data-sk]:hover,th.sortable:hover{color:var(--accent)}
.drow[hidden]{display:none}
.vwrap{overflow-x:auto;padding:2px 0 6px 12px}
button.tog{background:none;border:0;color:var(--accent);cursor:pointer;font-size:.85rem;padding:2px 0;white-space:nowrap}
#models td:nth-child(3){max-width:260px;overflow-wrap:anywhere}
.cmpcol{display:none}
#models.hascmp .cmpcol{display:table-cell}
button.copy{font-size:11px;padding:2px 7px;margin-top:4px}
.panel{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px 16px;align-items:end;
background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 14px}
.panel label{display:flex;flex-direction:column;gap:4px;font-size:.76rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.panel select,.panel input[type=number],.panel input[type=search]{width:100%;padding:8px 9px;border:1px solid var(--line);
border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;font-size:.95rem;text-transform:none;letter-spacing:0;font-weight:400}
.panel .checks{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;grid-column:1/-1;border-top:1px solid var(--line);padding-top:10px}
.panel .checks label{flex-direction:row;align-items:center;gap:6px;color:var(--fg);font-size:.9rem;font-weight:400;text-transform:none;letter-spacing:0}
.panel .checks .grow{flex:1}
button{font:inherit;font-size:.88rem;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
.facets{grid-column:1/-1;border-top:1px solid var(--line);padding-top:12px;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,640px),1fr));gap:12px 36px}
.frow{display:grid;grid-template-columns:96px 1fr;gap:8px;align-items:start}.frow.wide{grid-column:1/-1}.frow[hidden]{display:none}
.active{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 4px}.active:empty{display:none}
.chip.more{border-style:dashed;color:var(--accent)}
.flabel{font-size:.85rem;color:var(--muted);padding-top:5px;font-weight:600}
.fchips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border-radius:999px;padding:3px 11px;font-size:.88rem}
.chip b{font-weight:400;color:var(--muted);margin-left:7px;font-variant-numeric:tabular-nums}
.chip.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset;font-weight:600}
.chip.zero{opacity:.4}
.tablebox.tall{max-height:none;overflow:visible}
.tall thead th{position:sticky;top:0;background:var(--card);z-index:2;box-shadow:0 1px 0 var(--line)}
tr.click{cursor:pointer}
tr.click:hover td{background:var(--bg)}
tr.pick td{background:var(--v0b)}
td .nm{font-weight:600}
.dim{color:var(--muted);font-size:.85rem}
details summary{cursor:pointer;color:var(--accent);font-size:.85rem;white-space:nowrap}
.vbar{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:8px 0 2px}
.vbar .chip{padding:1px 9px;font-size:.8rem}
.sub-table{margin-top:8px;font-size:.82rem;width:auto;min-width:380px}
.sub-table td{padding:3px 10px}
.more{margin:12px 0;text-align:center}
.cnt{margin:6px 2px 10px;color:var(--muted);font-size:.9rem}
th.num{text-align:right}
"""

MAC_JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
let LIM=String(D.defaults.lim||'6');
const OVER=1.0, LIMIT=150;
const usableOf=ram=>LIM==='def'?ram*(ram<=36?2/3:0.75):Math.max(0,ram-(+LIM));
const FP=/(^|[^a-z0-9])(f32|f16|bf16|fp16|fp32)([^a-z0-9]|$)/i;
const PROV={hf:'Hugging Face',ollama:'Ollama',modelscope:'ModelScope'};
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtP=p=>(p<10?p.toFixed(1):Math.round(p))+'B';
const fmtN=n=>n.toLocaleString('en-US');
const fmtCtx=c=>(c>=1024?Math.round(c/1024)+'k':c)+' context';
const kv=(p,s,ctx,kb,sw)=>kb?kb*(sw?ctx*sw[1]+Math.min(ctx,sw[0])*(1-sw[1]):ctx)/1073741824:0.5*(p||s/0.57)*ctx/32768;
const PFEFF=0.3;
const fmtT=t=>t<10?t.toFixed(1)+' s':(t<120?Math.round(t)+' s':(t/60).toFixed(1)+' min');
function cmdOf(m,label,cx,ncpu){
  const lab=(label||'').split(' ')[0], moe=!ncpu?'':(ncpu==='all'?' --cpu-moe':' --n-cpu-moe '+ncpu);
  if(m.p==='ollama'){ const ref=lab.includes(':')?lab:((lab===''||lab==='latest')?m.n:m.n+':'+lab);
    return 'ollama pull '+ref+'\nOLLAMA_CONTEXT_LENGTH='+cx+' ollama run '+ref; }
  const plain=lab&&lab!=='?'&&!lab.includes('/')&&/^[A-Za-z0-9_.-]+$/.test(lab);
  if(m.p==='hf'){ if(plain&&/^(I?Q\d(_[A-Z0-9]+)*|NVFP4|MXFP4|BF16|F16|F32)$/i.test(lab)) return 'llama-server -hf '+m.n+':'+lab+' -c '+cx+' -ngl 99'+moe;
    return 'hf download '+m.n+' --include "*'+label+'*"\n# then: llama-server -m <file>.gguf -c '+cx+' -ngl 99'+moe; }
  return 'modelscope download --model '+m.n+' --include "*'+label+'*"\n# then: llama-server -m <file>.gguf -c '+cx+' -ngl 99'+moe;
}
const copyBtn=(m,label,cx,ncpu)=>'<button type="button" class="copy" data-cmd="'+esc(cmdOf(m,label,cx,ncpu))+'" title="Copy download/start command">copy command</button>';
const labelBits=l=>{const m=/(?:^|[^a-z0-9])i?q(\d)/i.exec(l); if(m) return +m[1]; if(/f32|fp32/i.test(l)) return 32; if(FP.test(l)) return 16; return null;};
const calcBits=(size,params)=>{if(!params||params<0.1) return null; const b=size*1.0737*8/params; return b>40?null:b;};
const bitsOf=(size,params,label)=>{const l=labelBits(label); return l!==null?l:calcBits(size,params);};
const dispLabel=l=>l==='?'?'(no quant label)':l;
const shortLab=l=>'<span title="'+esc(l)+'">'+esc(dispLabel(l).split('/').pop())+'</span>';
const NONTEXT=new Set(['Embedding','Image','Audio']);
const isNonText=m=>(m.u||[]).length>0&&m.u.every(u=>NONTEXT.has(u));
let current=[], shown=LIMIT;

const FAMILY_RULES=[
 ['Qwen', /qwen|qwq/i], ['Llama', /llama/i], ['Gemma', /gemma/i],
 ['Mistral', /mistral|mixtral|ministral|magistral|codestral|devstral|pixtral/i],
 ['DeepSeek', /deepseek/i], ['Phi', /(?:^|[^a-z])phi(?:[^a-z]|$)/i],
 ['GPT-OSS', /gpt[-_]?oss/i], ['GLM', /glm/i], ['Granite', /granite/i],
 ['OLMo', /olmo/i], ['Nemotron', /nemotron/i], ['SmolLM', /smollm/i],
 ['Command', /command[-_ ]?r|c4ai-command/i], ['LFM', /(?:^|[^a-z])lfm/i]
];
function familiesOf(name){
 const leaf=name.split('/').pop();
 const matches=FAMILY_RULES.filter(([,pattern])=>pattern.test(leaf)).map(([family])=>family);
 return matches.length?matches:['Other / unknown'];
}
const positiveInput=id=>{const x=Number($(id).value);return Number.isFinite(x)&&x>0?x:0;};
function variantLimits(row){
 const memory=positiveInput('maxmem'), speed=positiveInput('mintps'), context=positiveInput('minctx');
 return (!memory||row.need<=memory)&&(!speed||(row.tps!==null&&row.tps>=speed))&&
        (!context||(row.ctm>0&&row.ctm>=context));
}
D.models.forEach(m=>{m.families=familiesOf(m.n);m.moe=m.v.some(v=>v[3]<1); m.params=Math.max(0,...m.v.map(v=>v[2]||0)); if(m.params<0.1) m.params=null;});

/* Hardware: Mac presets (memory by reserve selection) or the own machine (fixed limits) */
function hwOf(i){
  const h=D.hw[i];
  if(h.k==='mac'){ const u=usableOf(h.ram); return {i,n:h.n+' \u00b7 '+h.ram+' GB',ram:h.ram,bw:h.bw,tf:h.tf||null,gpu:u,rcap:0,uni:true,mem:u,mac:true,star:!!(D.mine&&D.mine[0].toLowerCase()===h.n.toLowerCase()&&Math.abs(D.mine[1]-h.ram)<0.5)}; }
  if(h.k==='gpu'||h.k==='custom'){
    let vram=h.vram, ram=h.ram, bw=h.bw, tf=h.tf;
    if(h.k==='custom'){ vram=+$('cvram').value||0; ram=+$('cram').value||0; bw=+$('cbw').value||null; tf=+$('ctf').value||null; }
    const gr=D.defaults.gr, rr=D.defaults.rr;
    return {i,n:h.n+' \u00b7 '+vram+' GB VRAM',ram,bw,tf,gpu:Math.max(0,vram-gr),rcap:Math.max(0,ram-rr),uni:false,mem:vram,note:'VRAM minus '+gr+' GB, RAM minus '+rr+' GB',mac:false,star:false,custom:h.k==='custom'};
  }
  return {i,n:h.n+(h.uni?'':' \u00b7 '+h.vram+' GB VRAM'),ram:h.ram,bw:h.bw,tf:h.tf||null,gpu:h.gpu,rcap:h.rcap||0,uni:!!h.uni,mem:h.uni?h.gpu:h.vram,note:h.note,mac:false,star:true};
}
let H=hwOf(D.defaults.mac), lastUni=H.uni, H2=null;
const verdictOf=(need,hw)=>need<=hw.gpu?'g':(hw.uni?'x':(hw.gpu>0&&need<=hw.gpu+hw.rcap?'p':(need<=hw.rcap?'c':'x')));
/* same model as place() in Python: GPU full / MoE offload (experts partly in RAM) / Partial offload / CPU only */
function placeV(v,kvg,hw,eff){
  const size=v[1], frac=v[3], sp=v[7], need=size+kvg+OVER, rbw=(+$('rbw').value||D.defaults.rbw)*1e9/1073741824, bw=hw.bw?hw.bw*1e9/1073741824:0;
  if(need<=hw.gpu) return {need,vd:'g',tps:bw?eff*bw/Math.max(size*frac,0.1):null,ncpu:null};
  if(hw.uni) return {need,vd:'x',tps:null,ncpu:null};
  if(sp&&hw.gpu>0){
    const exp=size*sp[0], dense=size-exp, onCpu=sp[2]?need-hw.gpu:exp;
    if(dense+kvg+OVER<=hw.gpu&&onCpu<=exp&&onCpu<=hw.rcap){
      const c=onCpu/exp; let ncpu='all';
      if(sp[2]){ const n=Math.min(sp[2],Math.ceil(c*sp[2]-1e-9)); ncpu=n>=sp[2]?'all':String(n); }
      return {need,vd:'m',tps:bw?eff/((dense+exp*sp[1]*(1-c))/bw+exp*sp[1]*c/rbw):null,ncpu};
    }
  }
  if(hw.gpu>0&&need<=hw.gpu+hw.rcap){ const g=Math.min(1,Math.max(0,(hw.gpu-kvg-OVER)/size)), a=Math.max(size*frac,0.1);
    return {need,vd:'p',tps:bw?eff/(a*g/bw+a*(1-g)/rbw):null,ncpu:null}; }
  if(need<=hw.rcap) return {need,vd:'c',tps:eff*rbw/Math.max(size*frac,0.1),ncpu:null};
  return {need,vd:'x',tps:null,ncpu:null};
}
const VLAB={g:'GPU full',m:'MoE offload',p:'Partial offload',c:'CPU only',x:'Too large'};
const VBADGE={g:['v0','\u2714 GPU full'],m:['v4','\u25d1 MoE offload'],p:['v1','\u25d0 Partial offload'],c:['v2','\u25d4 CPU only'],x:['v3','\u2716 too large']};
const tierCtx=mem=>mem<24?4096:(mem<48?32768:262144);

function evalHW(hw,ctx,eff,minBits){
  const octx=$('octx').value;
  const pick=arr=>{const prac=arr.filter(r=>!r.fp), pool=prac.length?prac:arr; return pool.length?pool.reduce((a,b)=>b.size>a.size?b:a):null;};
  return D.models.map(m=>{
    const rows=m.v.map(v=>{
      const cx=(m.p==='ollama'&&octx==='std')?Math.min(tierCtx(hw.mem),v[5]||m.ct||32768):ctx;
      const kvg=kv(v[2],v[1],cx,v[4]||m.kb,v[4]?v[6]:m.sw), P=placeV(v,kvg,hw,eff), need=P.need, vd=P.vd, fit=vd==='g', bits=bitsOf(v[1],v[2],v[0]);
      const cb=calcBits(v[1],v[2]), bshow=cb!==null?cb:bits;
      const qc=qClass(v[0]), okb=!minBits||(bits!==null&&bits>=minBits), qt=qType(v[0]), okQc=!sel.quant.size||sel.quant.has(qc), okQt=!sel.qtype.size||sel.qtype.has(qt), okq=okb&&okQc&&okQt;
      return {label:v[0],size:v[1],params:v[2],frac:v[3],cx,need,vd,fit,bits,bshow,okq,okb,qc,qt,okQc,okQt,ctm:(v[5]||m.ct||0),fp:FP.test(v[0]),
              tps:P.tps, ncpu:P.ncpu,
              pf:(fit&&hw.tf)?PFEFF*hw.tf*1e12/(2*(v[2]||v[1]/0.57)*1e9*v[3]):null};
    });
    const ptok=+$('ptok').value||8192;
    rows.forEach(row=>{row.limitOK=variantLimits(row);row.okq=row.okq&&row.limitOK;row.ttft=row.pf?ptok/row.pf:null;});
    const ok=rows.filter(r=>r.okq);
    const best=pick(ok.filter(r=>r.fit));
    const alt=best?null:(pick(ok.filter(r=>r.vd==='m'))||pick(ok.filter(r=>r.vd==='p'))||pick(ok.filter(r=>r.vd==='c')));
    return {m,rows,best,alt,vd:best?'g':(alt?alt.vd:'x')};
  });
}

const SIZES=[['0-8','up to 8B'],['8-20','8\u201320B'],['20-40','20\u201340B'],['40-80','40\u201380B'],['80-100000','over 80B']];
const DLS=[['1000','1,000+'],['10000','10,000+'],['100000','100,000+'],['1000000','1M+']];
const UPDS=[['30','last 30 days'],['90','last 90 days'],['365','last year']];
const NOW=Date.now()/1000;
const QUALS=[['0','any'],['4','Q4 or better'],['5','Q5 or better'],['8','Q8']];
const USES=['Chat/Text','Code','Reasoning','Vision','Tools','Math','Embedding','Image','Audio','Uncensored'];
function sizeOK(p,s){ if(p===null) return false; const [a,b]=s.split('-').map(Number); return p>a&&p<=b; }
const fitOpts=()=>H.uni?[['g','fits completely'],['x','does not fit']]:['g','m','p','c','x'].map(k=>[k,VLAB[k]]);
const fitDefault=()=>H.uni?['g']:(D.defaults.only_fit?['g','m','p']:['g','m','p','c']);
const FACETS=[
  {id:'fit',label:'Verdict',multi:true,opts:fitOpts,test:(r,v)=>r.vd===v},
  {id:'quant',label:'Quantization',multi:true,opts:()=>QOPTS.filter(o=>D.models.some(m=>m.v.some(v=>qClass(v[0])===o[0]))),test:(r,v)=>r.rows.some(x=>x.qc===v&&x.okb&&x.limitOK&&x.okQt)},
  {id:'qtype',label:'Quant type',multi:true,opts:()=>qTypeOpts(),test:(r,v)=>r.rows.some(x=>x.qt===v&&x.okb&&x.limitOK&&x.okQc)},
  {id:'family',label:'Model family',multi:true,opts:()=>[...new Set(D.models.flatMap(m=>m.families))].sort().map(f=>[f,f]),test:(r,v)=>r.m.families.includes(v)},
  {id:'country',label:'Country',multi:true,opts:()=>{const cs=[...new Set(D.models.map(m=>m.co||'Unknown'))].sort((a,b)=>(a==='Unknown')-(b==='Unknown')||a.localeCompare(b)); return cs.map(c=>[c,c]);},test:(r,v)=>(r.m.co||'Unknown')===v},
  {id:'prov',label:'Provider',multi:true,opts:()=>Object.keys(PROV).filter(p=>D.models.some(m=>m.p===p)).map(p=>[p,PROV[p]]),test:(r,v)=>r.m.p===v},
  {id:'use',label:'Use case',multi:true,opts:()=>USES.filter(u=>D.models.some(m=>(m.u||[]).includes(u))).map(u=>[u,u]),test:(r,v)=>(r.m.u||[]).includes(v)},
  {id:'size',label:'Model size',multi:true,opts:()=>SIZES,test:(r,v)=>sizeOK(r.m.params,v)},
  {id:'arch',label:'Architecture',multi:true,adv:true,opts:()=>[['dense','dense'],['moe','MoE']],test:(r,v)=>v==='moe'?r.m.moe:!r.m.moe},
  {id:'dl',label:'Downloads',multi:false,adv:true,opts:()=>DLS,test:(r,v)=>(r.m.d||0)>=+v},
  {id:'upd',label:'Updated',multi:false,adv:true,opts:()=>D.models.some(m=>m.t)?UPDS:[],test:(r,v)=>(r.m.t||0)>=NOW-(+v)*86400},
  {id:'chg',label:'Since last run',multi:true,opts:()=>D.models.some(m=>m.ch)?[['new','new'],['upd','updated']]:[],test:(r,v)=>r.m.ch===v},
  {id:'cmp',label:'Comparison',multi:true,opts:()=>H2?[['both','fits on both'],['a','only on '+H.n],['b','only on '+H2.n]]:[],
   test:(r,v)=>{const a=!!r.best,b=!!(r.c2&&r.c2.best); return v==='both'?(a&&b):(v==='a'?(a&&!b):(!a&&b));}},
  {id:'qual',label:'Min. quantization bits',multi:false,adv:true,opts:()=>QUALS}
];
const sel={}; let fewer=false;
function resetSel(){ FACETS.forEach(F=>sel[F.id]=new Set()); fitDefault().forEach(x=>sel.fit.add(x)); sel.qual='0'; }
resetSel();
function filt(){ return {sort:$('sort').value, minBits:+sel.qual}; }
function passF(r,skip){
  if(!skip.includes('use')&&!sel.use.size&&isNonText(r.m)) return false;
  if(!r.rows.some(v=>v.okb&&v.limitOK&&(skip.includes('quant')||v.okQc)&&(skip.includes('qtype')||v.okQt))) return false;
  const q=$('q').value.trim().toLowerCase();
  if(q&&!(r.m.n+' '+PROV[r.m.p]+' '+r.m.families.join(' ')+' '+(r.m.co||'')+' '+(r.m.og||'')+' '+(r.m.u||[]).join(' ')+' '+r.rows.map(v=>v.label).join(' ')).toLowerCase().includes(q)) return false;
  for(const F of FACETS){
    if(F.id==='qual'||skip.includes(F.id)) continue;
    const s=sel[F.id]; if(s.size&&![...s].some(v=>F.test(r,v))) return false;
  }
  return true;
}
const MAXC=14, expandedF=new Set(); let showEmpty=false;
function renderFacets(resBy){
  const res=resBy[String(sel.qual)], act=[];
  $('facets').innerHTML=FACETS.map(F=>{
    const opts=F.opts(); if(!opts.length) return '';
    let items=opts.map(([v,t])=>{
      let n;
      if(F.id==='qual') n=resBy[v].filter(r=>passF(r,['fit'])&&r.best).length;
      else n=res.filter(r=>passF(r,[F.id])&&F.test(r,v)).length;
      const on=F.id==='qual'?sel.qual===v:sel[F.id].has(v);
      if(on&&F.id!=='qual') act.push([F,v,t]);
      return {v,t,n,on};
    });
    if(F.id!=='qual'&&!showEmpty) items=items.filter(x=>x.on||x.n>0);
    if(!items.length) return '';
    let more='';
    if(F.id!=='qual'&&items.length>MAXC){
      if(expandedF.has(F.id)) more='<button type="button" class="chip more" data-more="'+F.id+'">fewer</button>';
      else { const keep=items.filter((x,i)=>i<MAXC||x.on); more='<button type="button" class="chip more" data-more="'+F.id+'">+'+(items.length-keep.length)+' more</button>'; items=keep; }
    }
    const chips=items.map(x=>'<button type="button" class="chip'+(x.on?' on':'')+(x.n===0&&!x.on?' zero':'')+'" data-f="'+F.id+'" data-v="'+esc(x.v)+'">'+esc(x.t)+'<b>'+x.n+'</b></button>').join('')+more;
    return '<div class="frow'+(F.id==='qtype'||F.id==='family'?' wide':'')+'"'+(F.adv&&fewer?' hidden':'')+'><div class="flabel">'+F.label+'</div><div class="fchips">'+chips+'</div></div>';
  }).join('');
  $('active').innerHTML=act.length?'<span class="dim">Active filters:</span>'+act.map(([F,v,t])=>'<button type="button" class="chip on" data-f="'+F.id+'" data-v="'+esc(v)+'" title="Remove filter">'+esc(F.label)+': '+esc(t)+' ×</button>').join(''):'';
  $('fewer').textContent=fewer?'More filters':'Fewer filters';
}
const tpsTxt=v=>v.tps?'~'+Math.round(v.tps)+(v.frac<1?'*':''):'\u2013';
const card=(k,v,s)=>'<div class="card"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="s">'+s+'</div></div>';

const QOPTS=[['FP','BF16/F16/F32'],['Q8','Q8'],['Q6','Q6'],['Q5','Q5'],['Q4','Q4'],['Q3','Q3'],['Q2','Q2'],['Q1','Q1'],['?','not specified']];
const QT_RE=/(?:^|[^a-z0-9])((?:i?q\d(?:_(?:k|0|1|s|m|l|xl|xxl|xs|xxs|nl|p|\d))*)|nvfp4|mxfp4|bf16|f16|fp16|f32|fp32)(?![a-z0-9])/i;
const qType=l=>{ const m=QT_RE.exec(l); if(!m) return '?'; let t=m[1].toUpperCase(); if(t==='FP16') t='F16'; if(t==='FP32') t='F32'; return t; };
let _qto=null;
function qTypeOpts(){
  if(!_qto){
    const rank=t=>{const c=qClass(t); const i=['FP','Q8','Q6','Q5','Q4','Q3','Q2','Q1','?'].indexOf(c); return i<0?99:i;};
    const set=new Set(); D.models.forEach(m=>m.v.forEach(v=>set.add(qType(v[0]))));
    _qto=[...set].sort((a,b)=>rank(a)-rank(b)||a.localeCompare(b)).map(t=>[t,t==='?'?'not specified':t]);
  }
  return _qto;
}
const qClass=l=>{ if(FP.test(l)) return 'FP'; const m=/(?:^|[^a-z0-9])i?q(\d)/i.exec(l); return m?'Q'+m[1]:'?'; };

let vsort={k:'size',d:-1}, sdir=1;
const closedKeys=new Set();   /* variant tables are open by default; this remembers the ones the user closed */
const VK={label:v=>v.label.toLowerCase(),size:v=>v.size,need:v=>v.need,tps:v=>v.tps===null?-1:v.tps,
          ttft:v=>v.ttft===null?1e12:v.ttft,vd:v=>'gmpcx'.indexOf(v.vd)};
const VDIR={label:1,size:-1,need:1,tps:-1,ttft:1,vd:1};
const arrow=(on,d)=>on?(d>0?' \u25b2':' \u25bc'):'';
const vth=(k,t,num)=>'<th class="dim sortable'+(num?' num':'')+'" data-k="'+k+'" title="Sort by '+t+'">'+t+arrow(vsort.k===k,vsort.d)+'</th>';
/* Minimum hardware: smallest Mac / NVIDIA preset on which a variant with enough bits (and speed) fits completely */
const LADDER={mac:[],gpu:[]};
D.hw.forEach((h,i)=>{ if(h.k==='mac') LADDER.mac.push(i); else if(h.k==='gpu') LADDER.gpu.push(i); });
LADDER.mac.sort((a,b)=>D.hw[a].ram-D.hw[b].ram||(D.hw[a].bw||0)-(D.hw[b].bw||0));
LADDER.gpu.sort((a,b)=>D.hw[a].vram-D.hw[b].vram||(D.hw[a].bw||0)-(D.hw[b].bw||0));
const MINMEMO=new Map();
function minHW(m){
  const ctx=+$('ctx').value, eff=+$('eff').value||0.6, minT=positiveInput('mintps'), octx=$('octx').value, mb=D.defaults.mbits;
  const key=m.p+'|'+m.n+'|'+ctx+'|'+eff+'|'+minT+'|'+octx+'|'+LIM;
  if(MINMEMO.has(key)) return MINMEMO.get(key);
  const out={};
  for(const kind of ['mac','gpu']){
    out[kind]=null;
    for(const i of LADDER[kind]){
      const hw=hwOf(i); let best=null;
      for(const v of m.v){
        if(FP.test(v[0])) continue;
        const b=bitsOf(v[1],v[2],v[0]); if(mb&&(b===null||b<mb-0.05)) continue;
        const cx=(m.p==='ollama'&&octx==='std')?Math.min(tierCtx(hw.mem),v[5]||m.ct||32768):ctx;
        const need=v[1]+kv(v[2],v[1],cx,v[4]||m.kb,v[4]?v[6]:m.sw)+OVER; if(need>hw.gpu) continue;
        const tps=hw.bw?eff*(hw.bw*1e9/1073741824)/Math.max(v[1]*v[3],0.1):null;
        if(minT&&(tps===null||tps<minT)) continue;
        if(!best||v[1]>best.size) best={label:v[0],size:v[1],tps};
      }
      if(best){ out[kind]={hw,best}; break; }
    }
  }
  MINMEMO.set(key,out); return out;
}
function minHWLine(m){
  const o=minHW(m), mb=D.defaults.mbits, minT=positiveInput('mintps');
  const t=x=>x?'<strong>'+esc(x.hw.n)+'</strong> ('+esc(dispLabel(x.best.label))+', '+x.best.size.toFixed(1)+' GB'+(x.best.tps?', ~'+Math.round(x.best.tps)+' t/s':'')+')':'none of the presets';
  return '<p class="dim" style="margin:6px 0 8px">Minimum hardware ('+(mb?'\u2265 '+mb+' bits':'any quantization')+(minT?', \u2265 '+minT+' t/s':'')+', completely in memory): Mac '+t(o.mac)+' \u00b7 NVIDIA '+t(o.gpu)+'</p>';
}
function variantInner(r){
  const rows=r.rows.filter(v=>v.okq).sort((a,b)=>{const x=VK[vsort.k](a),y=VK[vsort.k](b); return (x<y?-1:(x>y?1:0))*vsort.d;});
  const body=rows.length?rows.map(v=>
    '<tr><td>'+esc(dispLabel(v.label))+(v.cx!==+$('ctx').value?' <span class="dim">'+fmtCtx(v.cx)+'</span>':'')+'</td><td class="num">'+v.size.toFixed(1)+' GB</td><td class="num">'+v.need.toFixed(1)+
    ' GB</td><td class="num">'+tpsTxt(v)+'</td><td class="num">'+(v.ttft?fmtT(v.ttft):'\u2013')+'</td><td><span class="badge '+VBADGE[v.vd][0]+'">'+VBADGE[v.vd][1]+
    '</span>'+(v.vd==='m'?'<div class="dim">'+(v.ncpu==='all'?'all expert layers in RAM':v.ncpu+' expert layers in RAM')+'</div>':'')+'</td><td>'+copyBtn(r.m,v.label,v.cx,v.ncpu)+'</td></tr>').join(''):'<tr><td colspan="7" class="dim">No variant matches the filter.</td></tr>';
  return minHWLine(r.m)+'<table class="sub-table"><thead><tr>'+vth('label','Variant')+vth('size','File',1)+vth('need','Need',1)+vth('tps','Tokens/s',1)+vth('ttft','First token',1)+vth('vd','Verdict')+'<th></th></tr></thead><tbody>'+body+'</tbody></table>';
}

function render(resetShown){
  if(resetShown) shown=LIMIT;
  H=hwOf(+$('mac').value);
  if(H.uni!==lastUni){ lastUni=H.uni; sel.fit=new Set(fitDefault()); }
  const ctx=+$('ctx').value, raw=+$('eff').value;
  const eff=Number.isFinite(raw)&&raw>0?Math.min(raw,1):D.defaults.eff, f=filt();
  $('eff').value=eff;
  $('uni').disabled=!H.mac;
  $('uni').title=H.mac?'Reserve within shared RAM':'Fixed memory budget supplied when generating this report';
  const resBy={}; ['0','4','5','8'].forEach(b=>resBy[b]=evalHW(H,ctx,eff,+b));
  H2=$('cmp').value!==''?hwOf(+$('cmp').value):null;
  if(H2&&H2.i===H.i) H2=null;
  if(H2) ['0','4','5','8'].forEach(b=>{const r2=evalHW(H2,ctx,eff,+b); resBy[b].forEach((r,i)=>r.c2=r2[i]);});
  $('models').classList.toggle('hascmp',!!H2);
  $('cmphead').textContent=H2?'vs '+H2.n:'Compare';
  $('custpanel').style.display=(H.custom||(H2&&H2.custom))?'contents':'none';
  const res=resBy[String(f.minBits)];
  const fitAll=res.filter(r=>passF(r,['fit'])&&r.best).length, base=res.filter(r=>passF(r,['fit'])).length;
  $('cards').innerHTML=card('Hardware',esc(H.n)+(H.star?' \u2605':''),'RAM '+H.ram+' GB \u00b7 bandwidth '+(H.bw?Math.round(H.bw)+' GB/s':'unknown'))+
    card(H.uni?'Usable unified memory':'Usable GPU memory',H.gpu.toFixed(1)+' GB',H.mac?(LIM==='def'?'macOS default limit (2/3 up to 36 GB, otherwise 3/4)':H.ram+' GB minus '+LIM+' GB for macOS'):esc(H.note||''))+
    (H.uni?'':card('RAM for offloading',H.rcap.toFixed(1)+' GB','usable alongside the GPU (MoE/partial offload, CPU only) \u00b7 '+(+$('rbw').value||D.defaults.rbw)+' GB/s'))+
    card('Matching models',fitAll+' of '+base,'with at least one variant completely in GPU memory')+
    card('Context / efficiency',fmtN(ctx)+' tokens','efficiency '+Math.round(eff*100)+' %')+
    (H2?card('Compared with',esc(H2.n),res.filter(r=>passF(r,['fit'])&&r.c2&&r.c2.best).length+' of '+base+' models fit completely there'):'');
  let list=res.filter(r=>passF(r,[]));
  renderFacets(resBy);
  const key=r=>r.best?1e6+r.best.size:(r.alt?r.alt.size:-1);
  const by={size:(a,b)=>key(b)-key(a),
            tps:(a,b)=>(b.best&&b.best.tps||-1)-(a.best&&a.best.tps||-1),
            params:(a,b)=>(b.m.params||0)-(a.m.params||0),
            name:(a,b)=>a.m.s.localeCompare(b.m.s),
            dl:(a,b)=>(b.m.d||0)-(a.m.d||0),
            upd:(a,b)=>(b.m.t||0)-(a.m.t||0)};
  list.sort(by[f.sort]); if(sdir<0) list.reverse();
  document.querySelectorAll('#models > thead th[data-sk]').forEach(h=>{h.textContent=h.dataset.t+arrow(h.dataset.sk===f.sort,(f.sort==='name'?1:-1)*sdir);});
  current=list;
  const maxTps=Math.max(1,...list.map(r=>r.best&&r.best.tps||0));
  const vis=list.slice(0,shown);
  $('models').tBodies[0].innerHTML=vis.map((r,i)=>{
    const b=r.best||r.alt, m=r.m, nv=r.rows.filter(v=>v.okq).length;
    const bits=b&&b.bshow?'<div class="dim">\u2248 '+b.bshow.toFixed(1)+' bit</div>':'';
    const tps=(b&&b.tps&&(r.best||b.vd==='m')?'<span class="bar"><i style="width:'+Math.min(100,Math.round(100*b.tps/maxTps))+'%"></i></span>'+tpsTxt(b)+(r.best?'':'<div class="dim">with offload</div>'):(b&&b.tps?'<span class="dim">'+tpsTxt(b)+' (offload)</span>':'\u2013'))+
      (r.best&&b.ttft?'<div class="dim">first token ~'+fmtT(b.ttft)+' @ '+Math.round((+$('ptok').value||8192)/1024)+'k</div>':'');
    const c2=r.c2, b2=c2&&(c2.best||c2.alt);
    const cmpCell=!H2?'':'<td class="cmpcol">'+(b2?esc(dispLabel(b2.label))+'<div class="dim">'+b2.size.toFixed(1)+' GB'+(b2.tps?' \u00b7 '+tpsTxt(b2)+' t/s':'')+'</div>'+(c2.best?'':'<div><span class="badge '+VBADGE[c2.alt.vd][0]+'">'+VBADGE[c2.alt.vd][1]+'</span></div>'):'<span class="badge v3">\u2716 does not fit</span>')+'</td>';
    const badge=r.best?'':(r.alt?'<div><span class="badge '+VBADGE[r.alt.vd][0]+'">'+VBADGE[r.alt.vd][1]+'</span></div>':'');
    return '<tr><td><div class="nm">'+esc(m.s)+(m.moe?'<span class="tag">MoE</span>':'')+'</div><div class="dim">'+PROV[m.p]+
      (m.d?' \u00b7 '+fmtN(m.d)+' downloads':'')+(m.co?' \u00b7 '+esc(m.co)+' ('+esc(m.og)+')':'')+'</div>'+
      (m.u&&m.u.length?'<div class="use">'+m.u.map(u=>'<span class="tag">'+esc(u)+'</span>').join('')+'</div>':'')+'</td>'+
      '<td class="num">'+(m.params?fmtP(m.params):'\u2013')+'</td>'+
      '<td>'+(b?shortLab(b.label)+bits+badge:'<span class="badge v3">\u2716 does not fit</span>')+'</td>'+
      '<td class="num">'+(b?b.size.toFixed(1)+' GB<div class="dim">need '+b.need.toFixed(1)+' GB'+(b.cx!==ctx?' \u00b7 '+fmtCtx(b.cx):'')+'</div>'+(b.ctm&&b.cx>b.ctm?'<div class="dim">\u26a0 model only up to '+fmtCtx(b.ctm)+'</div>':''):'\u2013')+'</td>'+
      '<td class="num">'+tps+'</td>'+cmpCell+
      '<td>'+(b?'<div>'+copyBtn(m,b.label,b.cx,b.ncpu)+'</div>':'')+'<button type="button" class="tog" data-i="'+i+'">'+(closedKeys.has(m.p+'|'+m.n)?'\u25b8 ':'\u25be ')+nv+(nv===1?' variant':' variants')+'</button></td></tr>'+
      '<tr class="drow" data-i="'+i+'" data-k="'+esc(m.p+'|'+m.n)+'"'+(closedKeys.has(m.p+'|'+m.n)?' hidden':'')+'><td colspan="7"><div class="vwrap">'+variantInner(r)+'</div></td></tr>';
  }).join('');
  if(!list.length) $('models').tBodies[0].innerHTML='<tr><td colspan="7" style="padding:28px;text-align:center">No variants match these filters. Try a higher memory budget, a lower speed or context requirement, or reset the filters. Unknown speed/context values are excluded when their filter is active.</td></tr>';
  const activeLimits=[positiveInput('maxmem')?'Memory ≤ '+positiveInput('maxmem')+' GiB':'',positiveInput('mintps')?'Speed ≥ '+positiveInput('mintps')+' tokens/s':'',positiveInput('minctx')?'Supported context ≥ '+fmtN(positiveInput('minctx'))+' tokens':''].filter(Boolean);
  $('filter-summary').textContent=activeLimits.join(' · ')||'No memory, speed or supported-context limits. Family detection is based on model names.';
  $('csv').disabled=!list.length;
  const nt=sel.use.size?0:D.models.filter(isNonText).length;
  $('count').textContent=fmtN(list.length)+' models shown'+(list.length>vis.length?' (first '+fmtN(vis.length)+')':'')+(nt?' \u00b7 '+nt+' non-text models hidden (choose Embedding, Image or Audio under Use case to show them)':'');
  $('more').innerHTML=list.length>vis.length?'<button id="morebtn">Show '+fmtN(Math.min(LIMIT,list.length-vis.length))+' more</button>':'';
  renderOverview(ctx,eff,f);
}

function renderOverview(ctx,eff,f){
  const cur=+$('mac').value;
  let h='';
  D.hw.forEach((_,i)=>{
    const hw=hwOf(i);
    const resA=evalHW(hw,ctx,eff,f.minBits).filter(r=>passF(r,['fit']));
    const resB=f.minBits>=4?resA:evalHW(hw,ctx,eff,4).filter(r=>passF(r,['fit']));
    const n=resA.filter(r=>r.best).length;
    let big=null;
    resB.forEach(r=>{const b=r.best; if(b&&b.params&&(!big||b.params>big.b.params||(b.params===big.b.params&&b.size>big.b.size))) big={r,b};});
    const t=big?esc(big.r.m.s)+' \u00b7 '+fmtP(big.b.params)+' '+esc(dispLabel(big.b.label))+
      '<div class="dim">'+big.b.size.toFixed(1)+' GB'+(big.b.tps?' \u00b7 '+tpsTxt(big.b)+' t/s':'')+'</div>':'\u2013';
    h+='<tr class="click'+(i===cur?' pick':'')+'" data-i="'+i+'"><td>'+esc(hw.n)+(hw.star?' \u2605':'')+'</td><td class="num">'+
       (hw.bw?Math.round(hw.bw):'?')+' GB/s</td><td class="num">'+n+' / '+resA.length+'</td><td>'+t+'</td></tr>';
  });
  $('macs').tBodies[0].innerHTML=h;
}

function exportCsv(){
  const rows=[['Compare hardware','Compare variant','Compare tokens/s','First token (s)','Command','Hardware','Model','Provider','Downloads','Use case','Model family','Country','Developer','Parameters (B)','Verdict','Variant','File GB','Need GB','Tokens/s','Evaluated context','Supported context','Memory limit GiB','Minimum tokens/s','Minimum supported context']];
  current.forEach(r=>{const b=r.best||r.alt; const c2=r.c2, b2=c2&&(c2.best||c2.alt); rows.push([H2?H2.n:'',b2?b2.label:'',b2&&b2.tps?Math.round(b2.tps):'',b&&b.ttft?b.ttft.toFixed(1):'',b?cmdOf(r.m,b.label,b.cx,b.ncpu).replace(/\n/g,' ; '):'',H.n,r.m.n,PROV[r.m.p],r.m.d||'',(r.m.u||[]).join(', '),r.m.families.join(', '),r.m.co||'',r.m.og||'',r.m.params||'',VLAB[r.vd],b?b.label:'',b?b.size.toFixed(1):'',b?b.need.toFixed(1):'',b&&b.tps?Math.round(b.tps):'',b?b.cx:'',b&&b.ctm?b.ctm:'',positiveInput('maxmem')||'',positiveInput('mintps')||'',positiveInput('minctx')||'']);});
  const csv=rows.map(r=>r.map(c=>'"'+String(typeof c==='string'&&/^[=+@-]/.test(c)?"'"+c:c).replace(/"/g,'""')+'"').join(';')).join('\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
  a.download='llm-comparison.csv'; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}

(function init(){
  const cu=D.hw.find(x=>x.k==='custom'); if(cu){ $('cvram').value=cu.vram||''; $('cram').value=cu.ram||''; $('cbw').value=cu.bw||''; $('ctf').value=cu.tf||''; }
  let h='',last=null;
  D.hw.forEach((m,i)=>{
    const grp=m.k==='own'?'My machine':(m.k==='gpu'?'NVIDIA GPUs (system RAM 64 GB)':(m.k==='custom'?'Custom':m.n));
    if(grp!==last){ if(last!==null) h+='</optgroup>'; h+='<optgroup label="'+esc(grp)+'">'; last=grp; }
    const o=hwOf(i);
    h+='<option value="'+i+'">'+esc(o.n)+(o.star?' \u2605':'')+'</option>';
  });
  $('mac').innerHTML=h+'</optgroup>'; $('mac').value=D.defaults.mac;
  $('cmp').innerHTML='<option value="">(none)</option>'+h+'</optgroup>'; $('cmp').value='';
  const pts=[2048,4096,8192,16384,32768,65536,131072]; if(!pts.includes(D.defaults.ptok)){pts.push(D.defaults.ptok); pts.sort((a,b)=>a-b);}
  $('ptok').innerHTML=pts.map(c=>'<option value="'+c+'">'+fmtN(c)+' tokens</option>').join(''); $('ptok').value=D.defaults.ptok||8192;
  const ctxs=[4096,8192,16384,32768,65536,131072];
  if(!ctxs.includes(D.defaults.ctx)){ctxs.push(D.defaults.ctx); ctxs.sort((a,b)=>a-b);}
  $('ctx').innerHTML=ctxs.map(c=>'<option value="'+c+'">'+fmtN(c)+' tokens</option>').join('');
  $('ctx').value=D.defaults.ctx; $('rbw').value=D.defaults.rbw; if(D.defaults.mintps) $('mintps').value=D.defaults.mintps; $('eff').value=D.defaults.eff; $('octx').value=D.defaults.octx||'std';
  const lims=['4','6','8','12']; if(LIM!=='def'&&!lims.includes(LIM)){lims.push(LIM); lims.sort((a,b)=>a-b);}
  $('uni').innerHTML=lims.map(l=>'<option value="'+l+'">RAM minus '+l+' GB for macOS</option>').join('')+'<option value="def">macOS default limit (2/3 or 3/4)</option>'; $('uni').value=LIM;
})();

const custLabel=()=>{ const i=D.hw.findIndex(x=>x.k==='custom'); if(i<0) return; const t=hwOf(i).n; ['mac','cmp'].forEach(s=>{ const o=$(s).querySelector('option[value="'+i+'"]'); if(o) o.textContent=t; }); };
['cvram','cram','cbw','ctf'].forEach(id=>$(id).addEventListener('input',()=>{ custLabel(); render(true); }));
['mac','cmp','ptok','ctx','octx','eff','rbw','sort','maxmem','mintps','minctx'].forEach(id=>$(id).addEventListener('change',()=>{ if(id==='sort') sdir=1; render(true); }));
$('uni').addEventListener('change',()=>{ LIM=$('uni').value; render(true); });
$('q').addEventListener('input',()=>render(true));
$('reset').addEventListener('click',()=>{ $('q').value=''; ['maxmem','mintps','minctx'].forEach(id=>$(id).value=''); $('sort').value='size'; sdir=1; resetSel(); render(true); });
$('fewer').addEventListener('click',()=>{ fewer=!fewer; render(false); });
const chipClick=e=>{
  const b=e.target.closest('.chip'); if(!b) return;
  if(b.dataset.more){ const f=b.dataset.more; if(expandedF.has(f)) expandedF.delete(f); else expandedF.add(f); render(false); return; }
  const id=b.dataset.f, v=b.dataset.v;
  if(id==='qual') sel.qual=v;
  else { const st=sel[id], F=FACETS.find(x=>x.id===id); if(st.has(v)) st.delete(v); else { if(!F.multi) st.clear(); st.add(v); } }
  render(true);
};
$('facets').addEventListener('click',chipClick); $('active').addEventListener('click',chipClick);
$('showempty').addEventListener('click',()=>{ showEmpty=!showEmpty; $('showempty').textContent=showEmpty?'Hide empty options':'Show empty options'; render(false); });
$('csv').addEventListener('click',exportCsv);
$('models').addEventListener('click',e=>{
  const b=e.target.closest('.copy'); if(!b) return;
  e.preventDefault();
  const text=b.dataset.cmd, done=()=>{const o=b.textContent; b.textContent='copied'; setTimeout(()=>b.textContent=o,1200);};
  const fallback=()=>{const t=document.createElement('textarea'); t.value=text; document.body.appendChild(t); t.select(); try{document.execCommand('copy');}catch(_){} t.remove(); done();};
  if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done,fallback); else fallback();
});
const setRow=(tr,open)=>{ tr.hidden=!open; if(open) closedKeys.delete(tr.dataset.k); else closedKeys.add(tr.dataset.k); const t=tr.previousElementSibling&&tr.previousElementSibling.querySelector('.tog'); if(t) t.textContent=(open?'\u25be ':'\u25b8 ')+t.textContent.slice(2); };
$('expand-variants').addEventListener('click',()=>document.querySelectorAll('#models tr.drow').forEach(tr=>setRow(tr,true)));
$('collapse-variants').addEventListener('click',()=>document.querySelectorAll('#models tr.drow').forEach(tr=>setRow(tr,false)));
$('models').addEventListener('click',e=>{
  const th=e.target.closest('.sub-table th[data-k]');
  if(th){ const k=th.dataset.k; if(vsort.k===k) vsort.d=-vsort.d; else vsort={k,d:VDIR[k]}; render(false); return; }
  const mh=e.target.closest('#models > thead th[data-sk]');
  if(mh){ const k=mh.dataset.sk; if($('sort').value===k) sdir=-sdir; else { $('sort').value=k; sdir=1; } render(true); return; }
  const t=e.target.closest('.tog'); if(!t) return; const tr=t.closest('tr').nextElementSibling; if(tr&&tr.classList.contains('drow')) setRow(tr,tr.hidden); });
$('more').addEventListener('click',e=>{ if(e.target.id==='morebtn'){shown+=LIMIT; render(false);} });
$('macs').tBodies[0].addEventListener('click',e=>{
  const tr=e.target.closest('tr'); if(!tr) return;
  $('mac').value=tr.dataset.i; render(false); window.scrollTo({top:0,behavior:'smooth'});
});
render(true);
"""


ORIGIN_RULES = [   # (regex on the lowercase repo name, developer of the base model, country); first match wins
    (r"deepseek", "DeepSeek", "China"),
    (r"qwen|qwq", "Alibaba (Qwen)", "China"),
    (r"(^|[-_/])(chat)?glm|thudm|zai-org", "Zhipu AI (GLM)", "China"),
    (r"kimi|moonshot", "Moonshot AI (Kimi)", "China"),
    (r"minimax", "MiniMax", "China"),
    (r"(^|[-_/])yi[-_.]\d|01-ai", "01.AI (Yi)", "China"),
    (r"internlm|internvl", "Shanghai AI Lab (InternLM)", "China"),
    (r"baichuan", "Baichuan", "China"),
    (r"ernie", "Baidu (ERNIE)", "China"),
    (r"hunyuan", "Tencent (Hunyuan)", "China"),
    (r"minicpm|openbmb", "OpenBMB (MiniCPM)", "China"),
    (r"mistral|mixtral|ministral|magistral|codestral|devstral|pixtral|voxtral|mathstral", "Mistral AI", "France"),
    (r"(?<!o)llama", "Meta (Llama)", "USA"),
    (r"gemma", "Google (Gemma)", "USA"),
    (r"(^|[-_/])phi(\d|[-_.]|$)|(^|[-_/])orca-?2", "Microsoft (Phi/Orca)", "USA"),
    (r"gpt-?oss", "OpenAI", "USA"),
    (r"granite", "IBM (Granite)", "USA"),
    (r"olmo|tulu|molmo|allenai", "Ai2 (OLMo)", "USA"),
    (r"nemotron", "NVIDIA (Nemotron)", "USA"),
    (r"smollm|huggingfacetb", "Hugging Face (SmolLM)", "USA"),
    (r"command-r|c4ai|(^|[-_/])aya([-_.]|$)|cohere", "Cohere", "Canada"),
    (r"(^|[-_/])lfm|liquidai", "Liquid AI (LFM)", "USA"),
    (r"dbrx", "Databricks", "USA"),
    (r"grok", "xAI", "USA"),
    (r"zamba|zyphra", "Zyphra", "USA"),
    (r"falcon|tiiuae", "TII (Falcon)", "UAE"),
    (r"(^|[-_/])jais([-_.]|$)", "Inception / MBZUAI (Jais)", "UAE"),
    (r"(^|[-_/])solar([-_.]|$)|upstage", "Upstage (Solar)", "South Korea"),
    (r"exaone|lgai", "LG AI Research (EXAONE)", "South Korea"),
    (r"stablelm|stable-code|stabilityai", "Stability AI", "UK"),
    (r"jamba|ai21", "AI21 Labs (Jamba)", "Israel"),
    (r"sarvam", "Sarvam AI", "India"),
    (r"apertus|swiss-ai", "Swiss AI (Apertus)", "Switzerland"),
    (r"sea-?lion", "AI Singapore (SEA-LION)", "Singapore"),
]
_ORIGIN_RE = [(re.compile(p), org, c) for p, org, c in ORIGIN_RULES]


def origin_of(name):
    """(developer, country) of the base model, derived from the repo name; (None, None) if unknown."""
    n = str(name).lower()
    for rx, org, country in _ORIGIN_RE:
        if rx.search(n):
            return org, country
    return None, None


def short_name(cand):
    n = cand.name.split("/")[-1]
    return re.sub(r"(?i)[-_]?gguf$", "", n)


def mac_matrix(cands, context, quant_filter, efficiency):
    """Per Mac configuration and model: best practical variant that fits completely in memory."""
    matrix = []
    for chip, rams in MAC_CONFIGS:
        for ram in rams:
            hw = Hardware(f"Apple {chip} (Unified Memory)", float(ram), float(ram), unified=True, gpu_cap_gb=unified_cap(ram))
            bw = chip_bandwidth(chip, ram)
            cells = [best_fit(evaluate(c, hw, context, quant_filter, bw, efficiency)) for c in cands]
            matrix.append((chip, ram, bw, cells))
    return matrix


def biggest_q4(cands, cells):
    """Largest model (by parameters) whose best variant has at least Q4 quality (about 4.3 bits/weight)."""
    best = None
    for c, r in zip(cands, cells):
        bits = quant_bits(r.label, r.size_gb, r.params_b) if r else None
        if r and r.params_b and bits is not None and bits >= 4:
            key = (r.params_b, r.size_gb)
            if best is None or key > best[0]:
                best = (key, c, r)
    return best


def write_html_matrix(path, args, cands, quant_filter, notes, mine, own=None):
    esc = html_lib.escape
    models = []
    for cand in cands:
        vs = [[v.label, round(v.size_gb, 2), v.params_b, round(frac, 3),
               (v.meta or {}).get("kv_bpt"), (v.meta or {}).get("ctx_train"), (v.meta or {}).get("swa"),
               (None if cand.provider == "ollama" else moe_split(cand, v, frac))]
              for v, frac in prepare_variants(cand, quant_filter)]
        if vs:
            models.append({"n": cand.name, "s": short_name(cand), "p": cand.provider,
                           "d": cand.downloads, "u": usage_of(cand), "og": origin_of(cand.name)[0], "co": origin_of(cand.name)[1],
                           "kb": cand.meta.get("kv_bpt"), "ct": cand.meta.get("ctx_train"), "sw": cand.meta.get("swa"),
                           "ch": CHANGES.get((cand.provider, cand.name)), "t": updated_epoch(cand), "v": vs})
    hwl = [{"k": "own", **own}] if own else []
    hwl += [{"k": "gpu", "n": n, "vram": v, "ram": 64, "bw": bw, "tf": tf or gpu_tflops(n.split("x ")[-1])}
            for n, v, bw, tf in GPU_PRESETS]
    cu = own if (own and not own.get("uni")) else {"vram": 24, "ram": 64, "bw": 1008, "tf": 165}
    hwl.append({"k": "custom", "n": "Custom machine", "vram": cu.get("vram"), "ram": cu.get("ram"), "bw": round(cu.get("bw") or 0) or None, "tf": cu.get("tf")})
    hwl += [{"k": "mac", "n": chip, "ram": ram, "bw": chip_bandwidth(chip, ram), "tf": chip_tflops(chip, ram)}
            for chip, rams in MAC_CONFIGS for ram in rams]
    default = 0 if own else next((i for i, m in enumerate(hwl)
                                  if mine and m["n"].lower() == mine[0].lower() and abs(m["ram"] - mine[1]) < 0.5), None)
    if default is None:
        default = next((i for i, m in enumerate(hwl) if m["n"] == "M4 Max" and m["ram"] == 64), 0)
    data = {"models": models, "hw": hwl, "mine": list(mine) if mine else None,
            "defaults": {"only_fit": bool(getattr(args, "only_fit", False)), "mac": default, "ctx": args.context, "eff": args.efficiency, "ptok": getattr(args, "prompt_tokens", 8192), "octx": SETTINGS["ollama_ctx"], "mbits": getattr(args, "min_bits", 4) or 0, "rbw": SETTINGS.get("ram_bw") or RAM_BW_DEFAULT, "mintps": getattr(args, "min_tps", None), "gr": GPU_RESERVE_GB[SETTINGS["os"] or host_os()], "rr": os_reserve(SETTINGS["os"] or host_os()), "lim": "def" if SETTINGS["mac_limit"] == "default" else f"{os_reserve('macos'):g}"}}
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    search_txt = esc(", ".join((args.search or []) + (args.repo or [])))
    out = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
           "<title>LLM Hardware Check</title><style>", HTML_CSS, MAC_CSS,
           "</style></head><body><div class=\"wrap\" style=\"max-width:min(96vw,2400px)\">",
           "<h1>LLM Hardware Check</h1>",
           f"<p class=\"sub\">Created {datetime.now().strftime('%Y-%m-%d %H:%M')} \u00b7 "
           f"{len(models)} models \u00b7 Search: {search_txt}</p>",
           "<div class=\"panel\">",
           "<label>Hardware (all Mac presets, including Ultra)<select id=\"mac\"></select></label>",
           "<label>Context<select id=\"ctx\"></select></label>",
           "<label>Context for Ollama<select id=\"octx\"><option value=\"std\">Ollama default (by memory)</option>"
           "<option value=\"same\">Use selected context</option></select></label>",
           "<label>Efficiency<input id=\"eff\" type=\"number\" min=\"0.01\" max=\"1\" step=\"any\"></label>",
           "<label>Prompt length (first-token time)<select id=\"ptok\"></select></label>",
           "<label>Compare with<select id=\"cmp\"></select></label>",
           "<label>Usable Mac memory<select id=\"uni\"></select></label>",
           "<label>RAM bandwidth for offloading (GB/s)<input id=\"rbw\" type=\"number\" min=\"1\" step=\"any\"></label>",
           "<div id=\"custpanel\" style=\"display:none\"><label>Custom: VRAM (GB)<input id=\"cvram\" type=\"number\" min=\"0\" step=\"any\"></label>"
           "<label>Custom: system RAM (GB)<input id=\"cram\" type=\"number\" min=\"0\" step=\"any\"></label>"
           "<label>Custom: bandwidth (GB/s)<input id=\"cbw\" type=\"number\" min=\"0\" step=\"any\"></label>"
           "<label>Custom: TFLOPS<input id=\"ctf\" type=\"number\" min=\"0\" step=\"any\"></label></div>",
           "<label>Search<input id=\"q\" type=\"search\" placeholder=\"Model name \u2026\"></label>",
           "<label>Sort by<select id=\"sort\"><option value=\"size\">largest variant</option>"
           "<option value=\"tps\">fastest</option><option value=\"params\">model size</option>"
           "<option value=\"dl\">most popular</option><option value=\"upd\">newest</option><option value=\"name\">Name</option></select></label>",
           "<label>Maximum total memory (GiB)<input id=\"maxmem\" type=\"number\" min=\"0\" step=\"any\" placeholder=\"No limit\" aria-describedby=\"limit-help\"></label>",
           "<label>Minimum speed (tokens/s)<input id=\"mintps\" type=\"number\" min=\"0\" step=\"any\" placeholder=\"No minimum\" aria-describedby=\"limit-help\"></label>",
           "<label>Minimum supported context<select id=\"minctx\" aria-describedby=\"limit-help\"><option value=\"\">Any / unknown</option>"
           "<option value=\"4096\">4,096 tokens</option><option value=\"8192\">8,192 tokens</option>"
           "<option value=\"16384\">16,384 tokens</option><option value=\"32768\">32,768 tokens</option>"
           "<option value=\"65536\">65,536 tokens</option><option value=\"131072\">131,072 tokens</option>"
           "<option value=\"262144\">262,144 tokens</option></select></label>",
           "<p id=\"limit-help\" class=\"dim\" style=\"grid-column:1/-1;margin:0\">All variant filters must match the same variant. "
           "Total memory includes weights, KV cache and runtime overhead. Speed is an estimate (offloaded variants are limited by the RAM bandwidth); "
           "unknown speed or supported context is excluded when its minimum is set. Supported context filters model metadata; "
           "the Context control above sets the context used for the memory calculation.</p>",
           "<div id=\"filter-summary\" role=\"status\" aria-live=\"polite\" class=\"dim\" style=\"grid-column:1/-1\"></div>",
           "<div class=\"active\" id=\"active\"></div><div class=\"facets\" id=\"facets\"></div>",
           "<div class=\"checks\"><button id=\"fewer\" type=\"button\">Fewer filters</button><button id=\"showempty\" type=\"button\">Show empty options</button>"
           "<button id=\"expand-variants\" type=\"button\">Expand all variants</button><button id=\"collapse-variants\" type=\"button\">Collapse all variants</button>"
           "<span class=\"grow\"></span><button id=\"reset\" type=\"button\">Reset filters</button>"
           "<button id=\"csv\" type=\"button\">Export as CSV</button></div></div>",
           "<div class=\"cards\" id=\"cards\"></div>",
           "<div class=\"cnt\" id=\"count\"></div>",
           "<div class=\"tablebox tall\"><table id=\"models\"><thead><tr><th data-sk=\"name\" data-t=\"Model\">Model<th class=\"num\" data-sk=\"params\" data-t=\"Parameters\">Parameters"
           "<th>Best variant<th class=\"num\" data-sk=\"size\" data-t=\"File\">File<th class=\"num\" data-sk=\"tps\" data-t=\"Tokens/s (approx.)\">Tokens/s (approx.)<th class=\"cmpcol\" id=\"cmphead\">Compare<th>Details</tr></thead>"
           "<tbody></tbody></table></div><div class=\"more\" id=\"more\"></div>",
           "<details class=\"fold\" style=\"margin-top:28px\"><summary><h2>All Mac configurations, including Ultra</h2></summary>"
           "<p class=\"dim\">Click a row to select that hardware above. The numbers follow the filters; "
           "the largest model in Q4 or higher bit depth is shown.</p>",
           "<div class=\"tablebox\"><table id=\"macs\"><thead><tr><th>Hardware<th class=\"num\">Bandwidth"
           "<th class=\"num\">Models<th>Largest model (Q4 or better)</tr></thead><tbody></tbody></table></div></details>"]
    if notes:
        out.append("<h2 style=\"margin-top:28px\">Notes</h2><ul class=\"notes\">"
                   + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>")
    out.append("<footer><p>Memory and file sizes are measured in GiB (shown as GB for compatibility). Bandwidth is measured in decimal GB/s. Quantization bit depth is not a quality benchmark.</p><p style=\"opacity:.6\">llm_hardware_check v" + __version__ + "</p><p><strong>Best variant</strong> = the largest variant that fits completely into usable memory "
               "(RAM minus the operating-system reserve, adjustable above); F16/BF16/F32 only if nothing else fits. "
               "<strong>Tokens/s</strong> is a rough upper bound for text generation: efficiency \u00d7 bandwidth "
               "\u00f7 active weights. MoE models (*) are detected from the GGUF expert count or the name (e.g. 30B-A3B, 8x22B); "
               "real values are usually clearly lower. <strong>First token</strong> = prompt length \u00f7 prompt speed "
               "(compute-bound: about 30 % of the GPU TFLOPS, \u00f7 2 FLOPs per active parameter and token). "
               "<strong>Country</strong> = home country of the developer of the base model, derived from the model name; fine-tunes and re-quantizations by third parties keep the origin of their base model. \"Unknown\" = no rule matches. "
               "<strong>Sliding-window models</strong> (Gemma, OLMo 3, gpt-oss \u2026) need less KV cache than full attention. Bandwidths are rounded manufacturer figures, the memory sizes "
               "typical configurations \u2013 not every combination exists in every device. Macs with 36 GB use "
               "the smaller GPU with less bandwidth on the Max chips.</p></footer>")
    out.append("</div><script id=\"data\" type=\"application/json\">" + data_json + "</script>")
    out.append("<script>" + MAC_JS + "</script></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))


def run_all_macs(args, cands, quant_filter, notes, mine, own=None):
    good = []
    for cand in cands:
        if cand.error:
            msg = f"[{PROVIDERS[cand.provider][0]}] {cand.name}: error - {cand.error}"
            print("\n" + msg)
            notes.append(msg)
        else:
            good.append(cand)
    if not good:
        print("\nNo models loaded.")
        return
    matrix = mac_matrix(good, args.context, quant_filter, args.efficiency)
    print(f"\n{'Mac configuration':<20}{'Bandwidth':>11}{'Models':>9}   Largest model in Q4 or higher bit depth")
    for chip, ram, bw, cells in matrix:
        n_fit = sum(1 for r in cells if r)
        name = f"{chip} {ram} GB"
        bw_txt = f"{bw:.0f} GB/s" if bw else "?"
        big = biggest_q4(good, cells)
        if big:
            _, c, r = big
            tps = f", ~{r.tps:.0f} t/s" if r.tps else ""
            top = f"{short_name(c)} {r.params_b:.0f}B {r.label} ({r.size_gb:.1f} GB{tps})"
        else:
            top = "-"
        print(f"{name:<20}{bw_txt:>11}{n_fit:>5}/{len(good):<3}  {top}")
    print(f"\nRow = one Mac, number = models with at least one fully fitting variant (memory minus {os_reserve('macos'):.0f} GB for macOS usable{', macOS default limit' if SETTINGS['mac_limit'] == 'default' else ''}).")
    if args.html:
        write_html_matrix(args.html, args, good, quant_filter, notes, mine, own)
        print(f"HTML report written: {os.path.abspath(args.html)}")


def advisor_catalog():
    """Hardware ladders for the advisor, smallest first: Macs by memory then bandwidth, NVIDIA presets by VRAM then bandwidth."""
    key = SETTINGS["os"] or host_os()
    macs = []
    for chip, rams in MAC_CONFIGS:
        for ram in rams:
            hw = Hardware(f"{chip} {ram} GB", float(ram), float(ram), unified=True, gpu_cap_gb=unified_cap(ram))
            macs.append((hw, chip_bandwidth(chip, ram), chip_tflops(chip, ram)))
    macs.sort(key=lambda t: (t[0].ram_gb, t[1] or 0))
    gpus = []
    for n, v, bw, tf in GPU_PRESETS:
        hw = Hardware(f"{n} ({v} GB)", float(v), 64.0, gpu_cap_gb=max(0.0, v - GPU_RESERVE_GB[key]),
                      ram_cap_gb=max(0.0, 64.0 - os_reserve(key)))
        gpus.append((hw, bw, tf or gpu_tflops(n.split("x ")[-1])))
    gpus.sort(key=lambda t: (t[0].vram_gb, t[1] or 0))
    return [("Mac", macs), ("NVIDIA", gpus)]


def advisor_pick(cand, hw, bw, args, quant_filter):
    """Largest variant with at least --min-bits that fits completely and reaches --min-tps (None if not)."""
    ok = []
    for r in evaluate(cand, hw, args.context, quant_filter, bw, args.efficiency):
        if r.verdict != "GPU full" or is_full_precision(r.label):
            continue
        bits = quant_bits(r.label, r.size_gb, r.params_b)
        if args.min_bits and (bits is None or bits < args.min_bits - 0.05):
            continue
        if args.min_tps and (r.tps is None or r.tps < args.min_tps):
            continue
        ok.append(r)
    return max(ok, key=lambda r: r.size_gb) if ok else None


def run_advisor(args, cands, quant_filter):
    """Reverse question: which hardware do I need at least for these models?"""
    good = [c for c in cands if not c.error]
    if not good:
        print("\nNo models loaded.")
        return
    cat = advisor_catalog()
    crit = [f"context {args.context}", f"at least {args.min_bits:g} bits/weight" if args.min_bits else "any quantization"]
    if args.min_tps:
        crit.append(f"at least {args.min_tps:g} tokens/s")
    print("\n=== Hardware advisor: smallest configuration per model ===")
    print("Criteria: completely in GPU/unified memory, " + ", ".join(crit) + f", efficiency {args.efficiency:.0%}")
    per = []
    for cand in good:
        res = {}
        for kind, ladder in cat:
            res[kind] = None
            for idx, (hw, bw, _) in enumerate(ladder):
                r = advisor_pick(cand, hw, bw, args, quant_filter)
                if r:
                    res[kind] = (idx, hw, r)
                    break
        per.append((cand, res))

    def cell(x):
        if not x:
            return "no preset meets the criteria"
        _, hw, r = x
        return f"{hw.gpu_name}: {r.label} {r.size_gb:.1f} GB" + (f", ~{r.tps:.0f} t/s" if r.tps else "")
    w = max(len(short_name(c)) for c, _ in per) + 2
    w = min(max(w, 12), 48)
    print(f"\n{'Model':<{w}}{'Smallest Mac':<58}Smallest NVIDIA preset")
    for cand, res in per:
        print(f"{short_name(cand)[:w-1]:<{w}}{cell(res['Mac'])[:57]:<58}{cell(res['NVIDIA'])}")
    print("\nAll models (one at a time):")
    for kind, ladder in cat:
        hits = [res[kind] for _, res in per]
        missing = [short_name(c) for (c, res) in per if not res[kind]]
        if all(h is None for h in hits):
            print(f"  {kind}: no preset is large enough")
            continue
        top = max((h for h in hits if h), key=lambda h: h[0])
        print(f"  {kind}: at least {top[1].gpu_name}"
              + (f" (no preset meets the criteria for: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})" if missing else ""))
    print("\nOrder = memory first, then bandwidth; no prices. NVIDIA presets assume 64 GB system RAM; "
          "for several cards the bandwidth of one card is used.")


FAMILIES = ["llama", "qwen", "gemma", "mistral", "deepseek", "phi", "gpt-oss", "glm", "granite",
            "olmo", "nemotron", "smollm", "command-r", "lfm"]


def main():
    ap = argparse.ArgumentParser(description="Check which LLMs from several providers fit your hardware")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--provider", choices=[*PROVIDERS, "all"], default=None,
                    help="Provider: hf (default), ollama, modelscope or all (with --all: all)")
    ap.add_argument("--all", "--top", dest="top", nargs="?", const=3, type=int, metavar="N",
                    help="Default view without --search: the most popular models (N per model family and provider, "
                         "default 3) from Hugging Face, Ollama and ModelScope; families: " + ", ".join(FAMILIES) +
                         ". Writes the HTML report and prints a compact overview")
    ap.add_argument("--search", action="append",
                    help="Search term, e.g. 'qwen' or 'llama 3'; repeatable: --search qwen --search gemma")
    ap.add_argument("--repo", action="append",
                    help="Specific model, optionally with prefix hf:/ollama:/ms: (repeatable)")
    ap.add_argument("--limit", type=int, default=10, help="Number of models per provider with --search (default 10)")
    ap.add_argument("--context", type=int, default=8192, help="Context length in tokens (default 8192)")
    ap.add_argument("--quant", help="Only these quants/tags, comma-separated, e.g. Q4_K_M,Q5_K_M")
    ap.add_argument("--only-fit", action="store_true", help="Only show variants that run on the GPU or partially offload")
    ap.add_argument("--vram", type=float, help="Set GPU memory budget in GiB (within shared RAM on Apple Silicon)")
    ap.add_argument("--ram", type=float, help="Set total RAM in GiB manually")
    ap.add_argument("--unified", action="store_true",
                    help="Force unified memory (Apple Silicon), e.g. --unified --ram 64")
    ap.add_argument("--mac", metavar="CHIP+GB",
                    help="Shorthand for Apple Silicon, e.g. --mac \"m4 max 64\" or --mac 64")
    ap.add_argument("--bandwidth", type=float, metavar="GB/s",
                    help="Memory bandwidth for the tokens/s estimate (from a table for Apple chips)")
    ap.add_argument("--efficiency", type=float, default=None,
                    help="Share of the bandwidth that is actually used (default: value stored by --calibrate "
                         "for this machine, otherwise 0.6)")
    ap.add_argument("--calibrate", action="append", metavar="SPEC=TOKENS_PER_S",
                    help="Calibrate the speed estimate with a measured value, e.g. "
                         "--calibrate ollama:qwen2.5:14b=28.5 or --calibrate hf:owner/repo@Q4_K_M=31 "
                         "(repeatable; the median per machine is stored and used from then on)")
    ap.add_argument("--tflops", type=float, metavar="TFLOPS",
                    help="GPU compute for the prompt-processing (prefill) estimate; "
                         "table values for Apple chips and some NVIDIA cards")
    ap.add_argument("--prompt-tokens", type=int, default=8192, metavar="N",
                    help="Prompt length for the time-to-first-token estimate (default 8192)")
    ap.add_argument("--live", action="store_true",
                    help="Use the currently free memory instead of RAM minus a fixed reserve")
    ap.add_argument("--diff", action="store_true",
                    help="Show models that are new or updated since the last run of the same query")
    ap.add_argument("--commands", action="store_true",
                    help="Print a download/start command for the best variant of each model")
    ap.add_argument("--html", nargs="?", const="llm-report.html", metavar="FILE",
                    help="File name of the HTML report (always created; default: llm-report.html, "
                         "with --all-macs mac-comparison.html)")
    ap.add_argument("--no-html", action="store_true", help="do not write an HTML report")
    ap.add_argument("--no-cache", action="store_true", help="do not cache provider responses")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and reload everything")
    ap.add_argument("--cache-ttl", type=float, default=24, metavar="HOURS",
                    help="Cache lifetime in hours (default 24)")
    ap.add_argument("--os", choices=["macos", "windows", "linux"], help="Operating system of the target machine (default: this machine)")
    ap.add_argument("--reserve", type=float, metavar="GB",
                    help="Memory used by the operating system (default: macOS 6, Windows 8, Linux 3 GB)")
    ap.add_argument("--mac-limit", choices=["reserve", "default"], default="reserve",
                    help="Apple Silicon: 'reserve' = RAM minus reserve (raise the GPU limit via 'sudo sysctl iogpu.wired_limit_mb=...'), "
                         "'default' = macOS default limit (2/3 up to 36 GB, otherwise 3/4)")
    ap.add_argument("--ollama-ctx", choices=["std", "same"], default="std",
                    help="Context for Ollama models: 'std' = Ollama's default by memory size (4k under 24 GiB, "
                         "32k from 24, 256k from 48; at most the model context), 'same' = like --context")
    ap.add_argument("--no-meta", action="store_true",
                    help="do not load GGUF metadata (KV cache and MoE are then only estimated)")
    ap.add_argument("--ca-bundle", metavar="FILE",
                    help="PEM file with CA certificates (company proxy, Burp/ZAP); default: system/Python store")
    ap.add_argument("--ram-bandwidth", type=float, metavar="GB/s",
                    help=f"System RAM bandwidth for offloading speeds (default {RAM_BW_DEFAULT:g}; dual-channel DDR5 about 60-90, "
                         "workstation/server with 4-12 channels 150-500)")
    ap.add_argument("--advisor", action="store_true",
                    help="Reverse question: smallest Mac and smallest NVIDIA preset each model runs on completely")
    ap.add_argument("--min-tps", type=float, metavar="T/S", help="Advisor/report: minimum tokens/s")
    ap.add_argument("--min-bits", type=float, default=4, metavar="BITS",
                    help="Advisor/report: minimum bits per weight (default 4 = Q4 or better, 0 = any)")
    ap.add_argument("--all-macs", action="store_true",
                    help="Compare models against all Mac chips and memory sizes (M1 to M5)")
    args = ap.parse_args()
    import math
    for name in ("context", "limit", "top", "ram", "bandwidth", "cache_ttl"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("vram", "reserve"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            ap.error(f"--{name} must be nonnegative and finite")
    if args.efficiency is not None and (not math.isfinite(args.efficiency) or not 0 < args.efficiency <= 1):
        ap.error("--efficiency must be greater than 0 and at most 1")
    for name in ("tflops", "prompt_tokens"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error(f"--{name.replace('_', '-')} must be positive and finite")
    global USE_META
    USE_META = not args.no_meta
    SETTINGS.update(os=args.os, reserve=args.reserve, mac_limit=args.mac_limit, ollama_ctx=args.ollama_ctx, live=args.live, ram_bw=args.ram_bandwidth)
    cache_setup(not args.no_cache, args.cache_ttl, args.refresh)
    ssl_setup(args.ca_bundle)
    if SSL_NOTE:
        print(f"TLS: {SSL_NOTE}")
    if args.no_html:
        args.html = None
    elif not args.html:
        args.html = "mac-comparison.html" if args.all_macs else "llm-report.html"

    chip = None
    if args.mac:
        args.unified = True
        num = re.search(r"(?<![A-Za-z0-9])(\d{1,4})(?!\d)", args.mac)
        if num:
            args.ram = float(num.group(1))
            if args.ram <= 0:
                ap.error("--mac memory size must be positive")
        elif platform.system() != "Darwin":
            ap.error("--mac needs the memory size, e.g. --mac \"m4 max 64\"")
        label = re.sub(r"(?<![A-Za-z0-9])\d{1,4}(?!\d)\s*(gb)?", "", args.mac, flags=re.I)
        label = re.sub(r"(?i)(?<=m\d)(?=(pro|max|ultra))", " ", label).strip(" -_")
        chip = label.title() if label else None

    if args.top:
        if not args.search and not args.repo:
            args.search = [""] + FAMILIES   # empty search = most popular of the provider, plus the most important families
        args.limit = args.top
    if not args.search and not args.repo and not args.calibrate:
        ap.error("Please specify --search, --repo or --all")

    if args.provider is None:
        args.provider = "all" if args.top else "hf"
    selected = list(PROVIDERS) if args.provider == "all" else [args.provider]
    default_provider = "hf" if args.provider == "all" else args.provider

    hw = detect_hardware(args.vram, args.ram, args.unified)
    if hw.unified and args.vram is not None and args.vram > hw.ram_gb:
        ap.error("--vram cannot exceed total RAM on unified-memory hardware")
    if hw.unified and not chip and platform.system() == "Darwin":
        chip = detect_mac_chip()
    if chip:
        hw.gpu_name = f"Apple {chip} (Unified Memory)"
    bandwidth, bw_source = None, ""
    if args.bandwidth:
        bandwidth, bw_source = args.bandwidth, "set manually"
    elif chip and chip_bandwidth(chip, hw.ram_gb):
        bandwidth, bw_source = chip_bandwidth(chip, hw.ram_gb), f"table value {chip}"
    if bandwidth is None and not hw.unified and gpu_bandwidth(hw.gpu_name):
        bandwidth, bw_source = gpu_bandwidth(hw.gpu_name), f"table value {hw.gpu_name}"
        if re.match(r"\d+x ", hw.gpu_name):
            bw_source += ", bandwidth of one card"
    tflops = args.tflops or (chip_tflops(chip, hw.ram_gb) if hw.unified else gpu_tflops(hw.gpu_name))
    key = hw_key(hw, chip)
    eff_explicit = args.efficiency is not None
    if args.calibrate:
        calibrate(args.calibrate, key, bandwidth, default_provider)
        if not args.search and not args.repo:
            return
    eff_note = ""
    if not eff_explicit:
        stored, n_samples = stored_efficiency(key)
        args.efficiency = stored or 0.6
        eff_note = f" (calibrated from {n_samples} measurements)" if stored else ""
    notes = []
    if args.all_macs:
        print("=== Comparison across all Mac configurations (M1 to M5) ===")
        print(f"Context: {args.context} tokens, efficiency {args.efficiency:.0%}{eff_note}")
        print(f"Provider: {', '.join(PROVIDERS[p][0] for p in selected)}")
    else:
        print("=== Hardware ===")
        if hw.unified:
            print(f"GPU : {hw.gpu_name}, usable {hw.gpu_cap_gb:.1f} GB")
        else:
            print(f"GPU : {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM, usable {hw.gpu_cap_gb:.1f} GB)")
        print(f"RAM : {hw.ram_gb:.1f} GB" + ("" if hw.unified else f" (usable {hw.ram_cap_gb:.1f} GB)"))
        print(f"Limits: {hw.note}")
        print(f"Context: {args.context} tokens")
        if bandwidth:
            print(f"Bandwidth: {bandwidth:.0f} GB/s ({bw_source}), efficiency {args.efficiency:.0%}{eff_note}")
            if tflops:
                print(f"Compute: ~{tflops:g} TFLOPS (prompt processing; time to first token for {args.prompt_tokens} tokens shown)")
        else:
            print("Bandwidth: unknown - pass --bandwidth GB/s for the tokens/s estimate")
        print(f"Provider: {', '.join(PROVIDERS[p][0] for p in selected)}")
        if hw.ram_gb <= 0:
            print("Note: RAM not detected, please pass --ram.")

    # candidate list: (provider, name, downloads)
    todo = []
    for r in args.repo or []:
        prefix, sep, rest = r.partition(":")
        if sep and prefix.lower() in PREFIXES and rest:
            todo.append((PREFIXES[prefix.lower()], rest, None, None))
        else:
            todo.append((default_provider, r, None, None))

    seen = {(t[0], t[1]) for t in todo}
    jobs = [(term, p) for term in args.search or [] for p in selected]

    def run_search(job):
        term, p = job
        try:
            return PROVIDERS[p][1](term, args.limit), None
        except Exception as exc:
            return [], exc

    with ThreadPoolExecutor(max_workers=6) as pool:   # result order is preserved
        found = list(pool.map(run_search, jobs))
    net_errors = [repr(exc) for _, exc in found if exc]
    for (term, p), (hits, exc) in zip(jobs, found):
        if exc:
            msg = f"[{PROVIDERS[p][0]}] search '{term}' failed: {exc}"
            print("\n" + msg)
            notes.append(msg)
            continue
        for name, dl, *rest in hits:
            if (p, name) not in seen:
                seen.add((p, name))
                todo.append((p, name, dl, rest[0] if rest else None))

    quant_filter = [q.strip().upper() for q in args.quant.split(",")] if args.quant else None

    done = [0]

    def load(t):
        c = load_candidate(*t)
        done[0] += 1
        if len(todo) > 8 and sys.stderr.isatty():
            print(f"\rLoading models {done[0]}/{len(todo)} ...", end="", file=sys.stderr, flush=True)
        return c

    with ThreadPoolExecutor(max_workers=6) as pool:
        cands = list(pool.map(load, todo))
    if len(todo) > 8 and sys.stderr.isatty():
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    hint = ssl_hint(net_errors + [c.error for c in cands if c.error])
    if hint:
        notes.append(hint)
        print("\n" + hint)
    cands, hidden = dedupe_mirrors(cands)
    if hidden:
        notes.append(f"{hidden} repeated provider/repository entries were hidden.")
        print(f"\n{hidden} repeated provider/repository entries hidden.")
    if SKIPPED:
        msg = (f"{len(SKIPPED)} GGUF variants were skipped (missing file size or incomplete shard set): "
               + "; ".join(f"{r or '?'}: {f}" for r, f in SKIPPED[:5]) + (" ..." if len(SKIPPED) > 5 else ""))
        notes.append(msg)
        print("\n" + msg)
    diff_msg = diff_snapshot(cands, snapshot_sig(selected, args.search, args.repo), args.diff)
    if diff_msg:
        notes.append(diff_msg)
    with_meta = sum(1 for c in cands if c.meta.get("kv_bpt") or any(v.meta for v in c.variants))
    if with_meta:
        moe_meta = sum(1 for c in cands if meta_active_fraction(c.meta) < 1.0)
        msg = (f"KV cache computed from the GGUF metadata for {with_meta} of {len(cands)} models "
               f"(Hugging Face, Ollama); the rest are roughly estimated."
               + (f" {moe_meta} MoE models detected via the expert count." if moe_meta else ""))
        notes.append(msg)
        print("\n" + msg)
    if CACHE.dir:
        print(f"Cache: {CACHE.hits} responses from the cache, {CACHE.misses} newly loaded ({CACHE.dir}; "
              f"--refresh reloads, --no-cache disables it)")

    mine = (chip, hw.ram_gb) if (chip and (args.mac or hw.unified)) else None
    preset = bool(mine) and any(c.lower() == chip.lower() and any(abs(r - hw.ram_gb) < 0.5 for r in rams)
                                for c, rams in MAC_CONFIGS)
    own = None
    # Explicit overrides must survive even when chip/RAM match a preset.
    if not (hw.unified and preset) or args.vram is not None or args.bandwidth is not None or args.live or args.tflops:
        own = {"n": (hw.gpu_name if not hw.unified else f"{chip or 'Custom machine'} {hw.ram_gb:g} GB") + (" (live)" if args.live else ""),
               "ram": round(hw.ram_gb, 1), "vram": round(hw.vram_gb, 1), "gpu": round(hw.gpu_cap_gb, 2),
               "rcap": 0 if hw.unified else round(hw.ram_cap_gb, 2), "bw": bandwidth, "tf": tflops, "uni": hw.unified,
               "note": hw.note}
    if args.advisor:
        run_advisor(args, cands, quant_filter)
        if args.html:
            write_html_matrix(args.html, args, [c for c in cands if not c.error], quant_filter, notes, mine, own)
            print(f"\nHTML report (minimum hardware per model in the details): {os.path.abspath(args.html)}")
        return
    if args.all_macs:
        run_all_macs(args, cands, quant_filter, notes, mine, own)
        return

    results = []
    shown = 0
    for cand in cands:
        if cand.error:
            msg = f"[{PROVIDERS[cand.provider][0]}] {cand.name}: error - {cand.error}"
            print("\n" + msg)
            notes.append(msg)
            continue
        rows = evaluate(cand, hw, args.context, quant_filter, bandwidth, args.efficiency, tflops)
        results.append((cand, rows))
        if args.top:
            shown += 1
        elif print_candidate(cand, rows, args.only_fit, show_tps=bool(bandwidth),
                             show_pf=args.prompt_tokens if tflops else None,
                             commands=args.commands, ctx=args.context):
            shown += 1
    if args.top:
        print_top(results, bool(bandwidth), args.only_fit, args.commands, args.context)
        if args.html:
            write_html_matrix(args.html, args, [c for c in cands if not c.error], quant_filter, notes, mine, own)
            print(f"\nHTML report (with filters and hardware selection): {os.path.abspath(args.html)}")
        return
    if shown == 0:
        print("\nNo matching variants found.")

    print("\nLegend: [++] GPU full | [+M] MoE offload (experts partly in RAM, llama.cpp --n-cpu-moe) | "
          "[+ ] Partial offload (slower) | [~ ] CPU only | [--] Too large")
    if not hw.unified:
        print(f"Offload speeds assume {SETTINGS.get('ram_bw') or RAM_BW_DEFAULT:g} GB/s RAM bandwidth (--ram-bandwidth).")
    print("Need = file size + estimated KV cache + about 1 GB overhead.")
    if bandwidth:
        print("Tokens/s = rough upper bound: efficiency x bandwidth / active weights (* = MoE, real values usually lower).")
    if args.html:
        write_html_matrix(args.html, args, [c for c in cands if not c.error], quant_filter, notes, mine, own)
        print(f"\nHTML report written: {os.path.abspath(args.html)}")


if __name__ == "__main__":
    main()
