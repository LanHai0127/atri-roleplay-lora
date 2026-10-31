# Atri Roleplay LoRA

Unofficial, non-commercial **亚托莉（ATRI）** roleplay LoRA for **Qwen2.5-7B-Instruct**.

This repository contains the training / inference / evaluation **toolkit** and the documentation.
The adapter weights are published as a **Release asset**. The ATRI game script, the full training
corpus and the base model are **not** included in this repository.

> ATRI -My Dear Moments- and its characters are copyrighted by their original rights holders.
> This is an unofficial fan project, not affiliated with or endorsed by them.
> No game script or raw corpus is redistributed here.

## Model

| Item | Value |
|---|---|
| Base | `Qwen2.5-7B-Instruct` |
| Adapter | v6.3, LoRA `r=16`, `lora_alpha=24` at inference (0.75 strength) |
| Training | QLoRA 4bit NF4, 2 epochs, 2,171 train / 154 valid samples |
| Final loss | `train_loss 2.499`, `eval_loss 2.133` |
| Hardware | 1 × RTX 4090 24 GB, 33 min |
| Character | 亚托莉 / YHN-04B-009, calls the user “花海先生” |

v6.3 adds two explicit boundaries on top of v6.2:

- **No general-assistant work**: no code, homework, papers, PPT, tables, translation, web search.
- **Professional knowledge boundary**: out-of-scope concepts are answered with
  “（歪头）……这是什么？我不懂。”, while everyday knowledge (tides, simple arithmetic) is still answered.

## Quick start

```bash
git clone https://github.com/LanHai0127/atri-roleplay-lora.git
cd atri-roleplay-lora
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt
```

Download the base model into `models/Qwen2.5-7B-Instruct` (ModelScope or Hugging Face), then
download the adapter from the Release page:

```bash
gh release download v6.3 -p atri-v6-3-lora.tar.gz
mkdir -p models/atri-v6-3-lora
tar -xzf atri-v6-3-lora.tar.gz -C models/atri-v6-3-lora
```

Chat:

```bash
python scripts/chat.py \
  --base models/Qwen2.5-7B-Instruct \
  --adapter models/atri-v6-3-lora \
  --load-4bit on
```

One-shot:

```bash
python scripts/chat.py --once "亚托莉，你在做什么？" \
  --base models/Qwen2.5-7B-Instruct \
  --adapter models/atri-v6-3-lora \
  --load-4bit on
```

4bit inference needs about 6–7 GB of VRAM. On a 24 GB card you can use `--load-4bit off`
for bf16, but 4bit is the tested default.

## Training

The release does **not** ship the corpus. Prepare a ChatML JSONL with the same schema as
`examples/sample.jsonl`:

```json
{"id": "sample-1", "messages": [{"role": "system", "content": "..."},
 {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "meta": {"scene": "...", "user_role": "花海"}}
```

Then:

```bash
python scripts/train_lora.py \
  --model models/Qwen2.5-7B-Instruct \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --out outputs/atri-v6-3-7b \
  --max-len 1024 --epochs 2 --lr 2e-5 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
  --batch 1 --grad-accum 8 --load-4bit on
```

The trainer computes loss only on assistant tokens and uses the shared ChatML template in
`scripts/common.py`.

## Evaluation

Prompts used for the 40-prompt and boundary checks are in `examples/`:

```bash
python scripts/eval.py \
  --base models/Qwen2.5-7B-Instruct \
  --adapter models/atri-v6-3-lora \
  --prompts examples/atri-v5-prompts.jsonl \
  --out-dir eval --load-4bit on \
  --temp 0.6 --rep-penalty 1.15 --no-repeat-ngram 3 --no-base
```

See [`docs/v6.3_边界修复报告.md`](docs/v6.3_边界修复报告.md) for the full evaluation,
including the 10 boundary prompts and the remaining failure cases.

## Inference-side guards

`scripts/common.py` contains deterministic guards that sit **in front of** the model:

- `safety_reply()`: self-harm / suicide risk → ask about immediate safety.
- `role_reply()`: code / homework / papers / translation / weather / news → refuse in character.
- `knowledge_reply()`: out-of-scope concepts → “这是什么？我不懂。”
- `clean_reply()` + `ensure_name_address()`: repair name drift such as “花海老师/亚托少佐/亚托丽”.

These guards are Python code, not part of the adapter. **They do not carry over to a GGUF /
llama.cpp / Ollama deployment**; add an equivalent wrapper if you need the same behavior there.

## Files

```
scripts/common.py          system prompt, ChatML, guards, cleanup
scripts/chat.py            interactive / one-shot chat
scripts/train_lora.py      QLoRA trainer
scripts/eval.py            base vs adapter evaluation
scripts/merge_lora.py      merge adapter into base
scripts/prepare_data.py    generic data compiler
scripts/check_env.py       environment check
scripts/fetch_model.py     chunked HF mirror downloader
cloud/                     Linux / cloud training wrappers
docs/                      training reports and data notes
examples/                  evaluation prompts (no game text)
```

## Limitations

- Not a perfect ATRI: some preference questions are still evasive, and “不会离开” can
  occasionally conflict with the “can’t promise forever” setting.
- The adapter is trained for Chinese roleplay; English and long technical answers are weaker.
- It is **not** a therapist, doctor, lawyer, or general assistant. The safety guard is a
  keyword-based first line, not a clinical risk assessment.
- Quantized GGUF versions may lose some of the boundary behavior; test before deploying.

## License

- **Code** in this repository: MIT (see `LICENSE`).
- **Adapter weights** in Releases: for personal, non-commercial research use only; see
  `ADAPTER_LICENSE.md` and `NOTICE.md`.
- **Base model**: Qwen2.5-7B-Instruct, Apache-2.0.
- **ATRI game text / characters**: all rights belong to the original rights holders; no game
  text is included in this repository.
