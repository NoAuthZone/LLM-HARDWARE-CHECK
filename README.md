# LLM-HARDWARE-CHECK

**Find out which local LLMs actually fit your machine — before you download 40 GB.**

`llm_hardware_check.py` is a single-file, dependency-free Python tool. It looks up models on
**Hugging Face**, **Ollama** and **ModelScope**, reads the real file sizes (and, where possible, the GGUF
metadata), estimates the memory each variant needs, and tells you whether it runs fully on your GPU, needs
partial offloading, runs on the CPU only, or is too large. It also estimates tokens/s and the time to the first
token, and writes an interactive, self-contained HTML report with filters.

- Python 3.9+, **standard library only** (no `pip install`)
- **Detects your hardware automatically** — Apple Silicon (M1–M5, unified memory), NVIDIA GPUs, CPU-only machines
- **`--all`**: one command checks the important models of every family across all three providers
- Compare **all Macs at once** (M1 … M5 Ultra, common memory sizes), NVIDIA presets (RTX 3090 … 8× H100) or two machines side by side
- **Hardware advisor**: the reverse question — which Mac or NVIDIA setup do I need at least for these models?
- Everything is an **estimate** — see [Accuracy and limitations](#accuracy-and-limitations)


Example files (hand-picked models with approximate sizes, not live data):
[`docs/example-report.html`](docs/example-report.html) (download and open in a browser) and
[`docs/example-cli-output.txt`](docs/example-cli-output.txt).

---

## Quick start

```bash
git clone https://github.com/NoAuthZone/LLM-HARDWARE-CHECK.git
cd LLM-HARDWARE-CHECK

# check the important models of all providers against THIS machine (hardware is detected automatically)
python3 llm_hardware_check.py --all
```

That is all. The tool detects your hardware, checks the models and prints a compact overview. The full result
goes to `llm-report.html` — one self-contained file without external requests, so it can be opened in any browser,
archived or shared.

### What `--all` checks

`--all` runs one check across **all three providers** (Hugging Face, Ollama, ModelScope). Per provider it takes the
most popular models overall plus the most popular ones of each major model family: Llama, Qwen, Gemma, Mistral,
DeepSeek, Phi, gpt-oss, GLM, Granite, OLMo, Nemotron, SmolLM, Command R and LFM. By default these are 3 models per
family and provider; `--all 10` takes 10 each and gives a much broader picture (more requests, longer first run).
"All" therefore means the relevant models of every family, not every one of the hundreds of thousands of uploads on
Hugging Face — use `--search` or `--repo` for anything else.

```bash
python3 llm_hardware_check.py --all              # 3 per family and provider
python3 llm_hardware_check.py --all 10           # broader: 10 per family and provider
python3 llm_hardware_check.py --all --only-fit   # hide everything that does not run
python3 llm_hardware_check.py --all --commands   # with download/start command for the best variant
```

Responses are cached for 24 hours, so a second run takes seconds.

### Automatic hardware detection

Without hardware options the tool checks the machine it runs on:

| System | What is detected | How |
|---|---|---|
| **Apple Silicon** (M1–M5) | Chip (e.g. M4 Max), unified memory, bandwidth and compute from the chip table | `sysctl` |
| **NVIDIA GPU** (Linux/Windows) | Card name, VRAM (all cards summed), bandwidth and compute from the card table (desktop, laptop, data center) | `nvidia-smi` |
| **System RAM** | Total memory; with `--live` the currently free memory (and free VRAM) | `os.sysconf` / Windows API; free memory via `/proc/meminfo`, `vm_stat`, Windows API |
| **Operating system** | Memory reserve for the OS: macOS 6 GB, Windows 8 GB, Linux 3 GB | Python `platform` |
| **No GPU** | RAM only: everything is rated *CPU only* | — |

The detected values are printed at the top of every run (`=== Hardware ===`), so you can check them. Override
anything with `--mac`, `--vram`, `--ram`, `--bandwidth`, `--tflops`, `--os` or `--reserve`, e.g. to check a machine
you do not own:

```bash
python3 llm_hardware_check.py --all --mac "m4 max 64"                 # a Mac you are planning to buy
python3 llm_hardware_check.py --all --vram 24 --ram 64 --bandwidth 1008  # a GPU you do not have
```

Not detected automatically: AMD and Intel GPUs (pass `--vram` and `--bandwidth`), the bandwidth of cards missing from
the table, and the RAM bandwidth for offloading (`--ram-bandwidth`, default 60 GB/s). With several NVIDIA cards the
bandwidth of one card is used, because the layers are split across them.

### More ways to start

```bash
python3 llm_hardware_check.py --provider all --search qwen --limit 10    # a search across all providers
python3 llm_hardware_check.py --repo ollama:qwen2.5:14b --context 16384  # one specific model
python3 llm_hardware_check.py --repo hf:bartowski/Qwen2.5-14B-Instruct-GGUF
```

---

## Examples

```bash
# NVIDIA RTX 4090 (24 GB VRAM) with 64 GB RAM, speed and prefill estimates
python3 llm_hardware_check.py --vram 24 --ram 64 --bandwidth 1008 --tflops 165 --search llama

# Only things that fit, with copy-paste commands for the best variant
python3 llm_hardware_check.py --mac "m4 max 64" --provider ollama --search qwen --only-fit --commands

# Compare every Mac configuration (M1 … M5), your own Mac is marked
python3 llm_hardware_check.py --all-macs --mac "m4 max 64" --provider all --search qwen --search gemma

# Use the currently free memory instead of a fixed OS reserve
python3 llm_hardware_check.py --live --search mistral

# 8 GB GPU + 32 GB RAM: which MoE models run with expert offloading, and how fast?
python3 llm_hardware_check.py --vram 8 --ram 32 --bandwidth 272 --ram-bandwidth 70 --search qwen3 --commands

# What is new since the last run of the same query?
python3 llm_hardware_check.py --all --diff

# Hardware advisor: smallest Mac / NVIDIA setup for these models (Q4 or better, at least 20 tokens/s)
python3 llm_hardware_check.py --advisor --repo ollama:qwen3:32b --repo ollama:gpt-oss:120b --min-tps 20

# A GPU you do not own (e.g. a 254 GB setup): pass VRAM, bandwidth and TFLOPS yourself.
# Without --bandwidth there is no tokens/s estimate; on a machine with an NVIDIA card, --vram keeps that card's
# bandwidth from the table. Alternative: the NVIDIA presets or "Custom machine" in the HTML report.
python3 llm_hardware_check.py --vram 254 --ram 128 --bandwidth 3350 --tflops 990 --all

# Calibrate the speed estimate with your own measurements (ollama run --verbose)
python3 llm_hardware_check.py --mac "m4 max 64" --calibrate ollama:qwen2.5:14b=28.5
```

---

## What the HTML report offers

- **Hardware selector** with every Mac preset (M1–M5, Pro/Max/Ultra), NVIDIA presets (RTX 3090/4090/5090, multi-GPU, RTX 6000 Ada, RTX PRO 6000, A100, H100, H200, 4×/8× H100) and your own machine (marked "(live)" with `--live`)
- **Custom machine**: enter VRAM, system RAM, bandwidth and TFLOPS directly in the page; the table updates while you type
- **Minimum hardware** per model in the details: the smallest Mac and the smallest NVIDIA preset that run it completely in memory (respects the minimum speed)
- **Side-by-side comparison** of two hardware configurations ("Compare with")
- **Filters** (with live counts): verdict, quantization class, exact quant type (Q4_K_M, IQ3_XS, MXFP4 …), model
  family, country of the developer, provider, use case (chat, code, reasoning, vision, tools, embedding …),
  model size, dense vs. MoE, downloads, last update, minimum quantization bits, "new since last run"
- **MoE offload** verdict with speed and the matching `--n-cpu-moe` command; RAM bandwidth adjustable in the page
- **Limits**: maximum total memory, minimum tokens/s, minimum supported context
- **Sortable** model table and variant tables (click the column headers)
- **Copy command** button per variant (`ollama pull …`, `llama-server -hf …`, `hf download …`)
- Context, Ollama context mode, efficiency, prompt length and the macOS memory reserve are adjustable in the page
- **CSV export** of the current view
- Non-text models (embedding, image, audio) are hidden by default and shown on demand

---

## Command line reference

| Option | Description |
|---|---|
| `--all [N]` (alias `--top`) | Default view without `--search`: the most popular models (N per family and provider, default 3) from all providers, as compact table + HTML report |
| `--search TERM` | Search term, repeatable |
| `--repo [hf:\|ollama:\|ms:]NAME` | Check a specific model, repeatable (`ollama:qwen2.5:14b`, `hf:owner/repo`) |
| `--provider {hf,ollama,modelscope,all}` | Provider (default `hf`; `all` with `--all`) |
| `--limit N` | Models per provider and search term (default 10) |
| `--context N` | Context length for the memory calculation (default 8192) |
| `--ollama-ctx {std,same}` | `std`: Ollama's default context by memory size (4k below 24 GiB, 32k from 24, 256k from 48; capped by the model's context); `same`: use `--context` |
| `--quant Q4_K_M,Q5_K_M` | Only these quantizations / tags |
| `--only-fit` | Hide variants that do not fit |
| `--mac "m4 max 64"` | Apple Silicon shorthand (chip + memory) |
| `--unified` / `--vram GiB` / `--ram GiB` | Manual hardware description |
| `--bandwidth GB/s` | Memory bandwidth for the tokens/s estimate (table for Apple chips and NVIDIA cards) |
| `--ram-bandwidth GB/s` | System RAM bandwidth for MoE/partial offload and CPU-only speeds (default 60; dual-channel DDR5 about 60–90, workstations/servers 150–500) |
| `--efficiency F` | Share of the bandwidth reached in practice (default 0.6, or the calibrated value) |
| `--calibrate SPEC=TPS` | Calibrate the efficiency with measured speeds, e.g. `ollama:qwen2.5:14b=28.5`, `hf:owner/repo@Q4_K_M=31` |
| `--tflops T` / `--prompt-tokens N` | GPU compute and prompt length for the time-to-first-token estimate (default 8192 tokens) |
| `--live` | Use currently free memory (RAM and NVIDIA VRAM) |
| `--diff` | List new/updated models since the last run of the same query |
| `--commands` | Print a download/start command for the best variant |
| `--os {macos,windows,linux}` / `--reserve GB` | Operating-system memory reserve (defaults: macOS 6, Windows 8, Linux 3 GB) |
| `--mac-limit {reserve,default}` | Apple Silicon: RAM minus reserve (needs a raised GPU limit) or the macOS default limit (2/3 up to 36 GB, else 3/4) |
| `--advisor` | Reverse question: smallest Mac and smallest NVIDIA preset per model, plus the configuration that runs all of them (one at a time) |
| `--min-tps T` | Advisor/report: minimum tokens/s |
| `--min-bits B` | Advisor/report: minimum bits per weight (default 4 = Q4 or better, 0 = any) |
| `--all-macs` | Compare against all Mac chips and memory sizes |
| `--html [FILE]` / `--no-html` | Name of the report (default `llm-report.html`) / do not write it |
| `--no-meta` | Do not read GGUF headers (KV cache and MoE are then only estimated) |
| `--ca-bundle FILE` | PEM file with CA certificates, e.g. for a company proxy or Burp/ZAP (certificate checks stay on) |
| `--no-cache` / `--refresh` / `--cache-ttl H` | Cache control (default: 24 h) |

Optional: set `HF_TOKEN` for private or gated Hugging Face repositories.

---

## How it works

**Memory need** (GiB) = file size + KV cache + 1 GiB runtime overhead

- **File size** comes from the provider API (Hugging Face / ModelScope file listings, Ollama registry manifests).
  Sharded GGUF files are summed; incomplete shard sets and files without a size are skipped.
- **KV cache** is computed from the GGUF header when it can be read (a small HTTP range request, 256 KiB first):
  `layers × KV heads × (key + value length) × 2 bytes × context`. Without metadata it falls back to a rough
  estimate. Models with **sliding-window attention** (Gemma 2/3/3n, OLMo 3, gpt-oss, Cohere 2) are calculated
  with their smaller cache.
- **Classification:** *GPU full* (fits into usable GPU / unified memory), *MoE offload* (expert weights partly in
  RAM, see below), *Partial offload* (whole layers spill into RAM), *CPU only*, or *Too large*. On unified memory
  only *GPU full* and *Too large* exist.
- **Usable memory:** Apple Silicon: RAM minus the OS reserve (or the macOS default GPU limit). Discrete GPU:
  VRAM minus a small driver/desktop reserve, plus RAM minus the OS reserve for offloading.

**Tokens/s** ≈ `efficiency × memory bandwidth ÷ active weights`. Token generation is bandwidth-bound. For MoE
models only the active experts count; they are detected from the GGUF expert count or the model name
(`A3B`, `8x7B`). The estimate is an upper bound; MoE models in particular are often slower in practice.

**MoE offload** (llama.cpp `--n-cpu-moe N` / `--cpu-moe`): in a mixture-of-experts model almost all weights are
expert weights (Qwen3-30B-A3B about 95 %, gpt-oss-120b about 98 %), but each token uses only a few experts.
Attention, shared layers and the KV cache stay in VRAM together with as many expert layers as fit; the remaining
experts go to system RAM, from which only the active experts are read per token. That is why a 100B-class MoE model
runs at usable speed with a 24 GB GPU and enough RAM. The tool computes the expert share from the GGUF header
(experts × 3 × embedding × expert FFN × MoE layers; without header data from the active share in the name), how many
expert layers have to move (the `--n-cpu-moe` value in the command) and the speed:

`time per token ≈ (dense + active experts on the GPU) ÷ GPU bandwidth + active experts in RAM ÷ RAM bandwidth`

Partial and CPU-only offloading get a speed estimate too (layer split, read through the RAM bandwidth), so the
difference to MoE offloading is visible. The RAM bandwidth defaults to 60 GB/s (dual-channel DDR5) and is set with
`--ram-bandwidth` or in the report. Ollama models are only offloaded layer by layer; the advisor still asks for
"completely in memory".

**Time to first token** ≈ `prompt tokens ÷ prompt speed`, with the prompt speed roughly
`0.3 × TFLOPS ÷ (2 × active parameters)`. Compute values are rounded table figures (Apple chips, a few NVIDIA
cards) or `--tflops`.

**Advisor:** for every model the tool walks two ladders — all Mac presets (by memory, then bandwidth) and the
NVIDIA presets (by VRAM, then bandwidth; 64 GB system RAM assumed) — and reports the first configuration on which a
variant with at least `--min-bits` fits completely and reaches `--min-tps`. "Smallest" means least memory, not
cheapest: the tool has no prices.

**Calibration:** `--calibrate` turns your measured tokens/s into an efficiency factor. The median per machine is
stored and used automatically from then on.

**Country** in the report is the home country of the developer of the *base* model, derived from the model name
via a built-in table. Fine-tunes and re-quantizations keep the origin of their base model; unknown names are
listed as "Unknown".

---

## Files the tool writes

| Path | Content |
|---|---|
| `~/.cache/llm-hardware-check/` | Provider responses, 24 h (never requests that carry a token) |
| `~/.local/state/llm-hardware-check/` | `calibration.json` and the snapshots used by `--diff` |
| `./llm-report.html` | The report (current directory) |

---

## Troubleshooting: SSL / certificate errors

`CERTIFICATE_VERIFY_FAILED` means Python could not verify the provider's certificate. The tool prints one hint
instead of an error per model. Certificate checks are never switched off.

- **Python from python.org on macOS** ships without CA certificates. The tool then falls back to `certifi` (if
  installed) or the macOS system root certificates automatically. To fix it for good, run
  `open "/Applications/Python 3.x/Install Certificates.command"` (with your version, e.g. 3.15).
- **Intercepting proxy** (Burp, ZAP, company proxy) in `HTTPS_PROXY`: unset it for the run
  (`env -u HTTPS_PROXY -u https_proxy python3 llm_hardware_check.py …`) or pass its CA with `--ca-bundle ca.pem`.
- **Linux:** update the CA store (`sudo apt install --reinstall ca-certificates`).

## Accuracy and limitations

- All numbers are **estimates**. Real memory use depends on the runtime (llama.cpp, Ollama, LM Studio, MLX),
  its flags (flash attention, KV cache quantization, batch size) and the OS.
- Bandwidths and TFLOPS are rounded manufacturer figures; the M5 compute values are rough assumptions.
- Chip/memory combinations in the Mac presets are typical configurations, not every combination Apple sells.
- Ollama and ModelScope have no officially documented search API. If a search fails, use `--repo`.
- Country and use-case labels are heuristics based on names and provider metadata.
- Duplicate repositories (mirrors, re-uploads) are **not** merged: equal names and sizes do not prove equal files.
- Only GGUF-based models are covered (no MLX or safetensors-only repositories).

---

## Contributing

Issues and pull requests are welcome, especially for new chip/GPU table entries, more developer/country rules,
and provider-API changes. The code is a single readable file; there are no build steps.

## License

Add a license file of your choice (for example MIT) before publishing.
