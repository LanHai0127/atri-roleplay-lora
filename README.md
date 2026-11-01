# Atri Roleplay LoRA

Unofficial, non-commercial **亚托莉（ATRI）** roleplay LoRA for **Qwen2.5-7B-Instruct**.

This repository contains the training / inference / deployment **toolkit** and the documentation.
The adapter weights are published as a **Release asset**. The ATRI game script, the full training
corpus and the base model are **not** included in this repository.

> ATRI -My Dear Moments- and its characters are copyrighted by their original rights holders.
> This is an unofficial fan project, not affiliated with or endorsed by them.
> No game script or raw corpus is redistributed here.

## Model

| Item | v7 | v6.3 (previous) |
|---|---|---|
| Base | `Qwen2.5-7B-Instruct` | same |
| Adapter | LoRA `r=32`, `alpha=64`, 7 projections | `r=16`, `alpha=24` at inference |
| Training | QLoRA 4bit NF4, 3 epochs, **22,493** train / 686 valid | 2 epochs, 2,171 / 154 |
| Final loss | `0.855` | `2.499` |
| Reply length | **median 11 / p90 24 chars** | tens of chars, often 3 paragraphs |
| System prompt | **none — 100% of the corpus is system-free** | every sample carried one |
| One run | 1 × RTX 4090, 44.6 min, peak 19.4 GB | 33 min, peak 9.9 GB |

v7 is the "she stops talking" release. Three things changed:

1. **No system prompt anywhere.** v6's 2,171 samples all carried one, so the model learned
   "map this system prompt to this persona" instead of *being* the persona. That is the root
   cause of issue #1 (falling back to a generic assistant).
2. **Short replies.** `你好` now gets 4–9 characters, not a paragraph.
3. **Long input handled.** The corpus was extended with long confessions (up to 123 chars).

## What went wrong in v6.3 (and why v7 took four runs)

The first v7 run passed 4 of 26 probes with `train_loss 1.595`. Three real bugs, found in order:

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Effective LR diluted 4× — token-budget batching raised the batch from 16 to 71 samples, but `lr` stayed at `5e-5` | Could not even reproduce an 11-char answer from the training set (output 36 chars) | `lr 2e-4`, `accum 1` |
| 2 | **`<|im_end|>` never entered the labels** — only the assistant body was supervised | `loss 0.852` and still writing 90 chars; it never learned *where to stop* | include the trailing `<|im_end|>` (27,298 occurrences) in the label span |
| 3 | Multi-turn "repeat greeting" lines leaked into single-turn samples | `你好` answered "didn't we just talk about this?" 4 times out of 4 | dropped 86 samples, hand-wrote 384 greeting pairs |

Bug 2 is the one worth remembering: with a low loss and endless output, **check the label
span before touching sampling parameters**.

Full write-up: [`docs/v7_训练报告.md`](docs/v7_训练报告.md) · corpus notes:
[`docs/v7_语料说明.md`](docs/v7_语料说明.md)

## Training history (read this before quoting a duration)

**44.6 min** is only the net time of the final v7 run (894 steps). The project was not trained
once for 44 minutes. There were:

- v5 full run, v6-clean, v6.1, v6.2, v6.3;
- **four** v7 runs — three of them discarded (LR, missing `<|im_end|>`, polluted greetings);
- 7B smoke tests, batch-size probes, memory tests, evaluation passes between runs.

**Cumulative 4090 training time is now over 7 hours.** The numbers in the reports are per-run,
not the total project cost.

## Quick start

```bash
git clone https://github.com/LanHai0127/atri-roleplay-lora.git
cd atri-roleplay-lora
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt
```

Download the base model into `models/Qwen2.5-7B-Instruct`, then the adapter from Releases:

```bash
gh release download v7 -p atri-v7-lora.tar.gz
mkdir -p models/atri-v7-lora
tar -xzf atri-v7-lora.tar.gz -C models/atri-v7-lora
```

Chat (**use `chat_v7.py`, not `chat.py`** — the v6 script injects a system prompt):

```bash
python scripts/chat_v7.py \
  --base models/Qwen2.5-7B-Instruct \
  --adapter models/atri-v7-lora \
  --load-4bit on
```

One-shot:

```bash
python scripts/chat_v7.py --once "我好累" \
  --base models/Qwen2.5-7B-Instruct --adapter models/atri-v7-lora --load-4bit on
```

4bit inference needs about 6–7 GB of VRAM.

### Two rules for v7 inference

Both are enforced in `scripts/chat_v7.py` and in `deploy/llama.cpp/chat-template.jinja`:

1. **Never add a system prompt.** The persona lives in the weights. A system message pushes the
   model back into assistant mode.
2. **Prefix user content with `花海：`.** Every training sample uses this format; skipping it is
   a train/inference distribution mismatch (replies get longer and more polite).

## Training

The release does **not** ship the corpus. Prepare a system-free ChatML JSONL:

```json
{"messages": [
  {"role": "user", "content": "花海：今天有点累"},
  {"role": "assistant", "content": "（坐在旁边）嗯。那就先休息一会儿吧。"}
 ], "weight": 1.0}
```

```bash
python scripts/train_v7.py \
  --data data/train.jsonl --val data/valid.jsonl \
  --model models/Qwen2.5-7B-Instruct --out outputs/atri-v7-lora \
  --epochs 3 --accum 1 --gc --token-budget 4500 --lr 2e-4
```

Notes that cost real GPU hours:

- **Batch by token budget, not by sample count.** Lengths in this corpus vary 14×; a fixed
  `batch_size` is either wasteful or an OOM. `--token-budget 4500` → 19.4 GB peak on a 24 GB card.
- **Don't probe with 20 steps and declare victory.** bs=24 passed a 20-step probe and OOM'd at
  step 384, because the probe never sampled the longest group.
- `lr` is tied to the *effective* batch, which token-budget batching changes. See bug 1 above.

## Evaluation

```bash
python scripts/eval.py \
  --base models/Qwen2.5-7B-Instruct --adapter models/atri-v7-lora \
  --prompts examples/atri-v7-prompts.jsonl --out-dir eval --load-4bit on \
  --temp 0.75 --rep-penalty 1.1 --no-base
```

The 26 prompts in `examples/atri-v7-prompts.jsonl` cover persona (4), high-frequency greetings
(9), companionship (5), boundaries (4), relationship (2) and memory (2). v7 result: **26/26 pass**,
median 11 chars.

## Deployment

Pure-CPU / GGUF deployment via llama.cpp (Q8_0 base + runtime LoRA, no merging):

[`deploy/README.md`](deploy/README.md)

- Why not ollama: llama.cpp supports DRY sampling and `--lora` mounting.
- Measured on a Ryzen 5 4600U: **2.2 tok/s** generation, prefill 16.5 s cold / **1.0 s with prefix
  cache hit**. Prefix caching is what makes a 16k context usable on CPU — so don't let the caller
  inject per-message variable text (timestamps, random nicknames) into the prompt.

## Files

```
scripts/chat_v7.py         v7 chat (no system prompt, 花海： prefix)   ← use this
scripts/chat.py            v6 chat (injects system prompt) — kept for v6.3
scripts/train_v7.py        v7 QLoRA trainer (token-budget batching, im_end labels)
scripts/train_lora.py      v6 trainer
scripts/eval.py            base vs adapter evaluation
scripts/common.py          v6 system prompt, ChatML, guards, cleanup
scripts/merge_lora.py      merge adapter into base
scripts/prepare_data.py    generic data compiler
scripts/check_env.py       environment check
scripts/fetch_model.py     chunked HF mirror downloader
deploy/llama.cpp/          chat template, start script, systemd unit
examples/                  evaluation prompts (no game text)
docs/                      training reports and corpus notes
cloud/                     Linux / cloud training wrappers
```

## Limitations

- **Still not a perfect ATRI.** `算了不说了` occasionally degenerates into `嗯嗯～～～………………再聊`
  — tilde-plus-ellipsis is a residual pattern in the corpus.
- `在吗` answers `喂，不是刚问过嘛` roughly 1 time in 5 — leftover multi-turn contamination.
- `夸夸我` sometimes invents a drawing that was never made.
- Very short replies (X-class) are only 106 samples / 0.3% of the corpus, so the shortest register
  is the thinnest.
- Chinese-only. English and long technical answers are weak.
- This is **not** a therapist, doctor, lawyer, or general assistant. Any safety guard is a
  keyword-based first line, not a clinical risk assessment.
- The v6 keyword guards in `scripts/common.py` do **not** carry over to a GGUF / llama.cpp
  deployment. v7 does not rely on them: the refusals above come from the weights.

## License

- **Code** in this repository: MIT (see `LICENSE`).
- **Adapter weights** in Releases: for personal, non-commercial research use only; see
  `ADAPTER_LICENSE.md` and `NOTICE.md`.
- **Base model**: Qwen2.5-7B-Instruct, Apache-2.0.
- **ATRI game text / characters**: all rights belong to the original rights holders; no game
  text is included in this repository.
