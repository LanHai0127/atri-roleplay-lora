# Model Card: Atri v7 LoRA

## Model details

- **Base model:** Qwen2.5-7B-Instruct (Apache-2.0)
- **Adapter:** LoRA, rank 32, alpha 64, on q/k/v/o/gate/up/down projections (323 MB)
- **Task:** Chinese character roleplay as 亚托莉 (ATRI, YHN-04B-009)
- **Training:** QLoRA 4bit NF4, 3 epochs, 22,493 train / 686 valid ChatML samples, **no system prompt**
- **Hardware:** 1 × RTX 4090 24 GB, 44.6 min for the final run, peak 19.4 GB
- **Final metrics:** train_loss 0.855, 26/26 probe pass, reply length median 11 / p90 24 chars

Previous release: v6.3 (`r=16`, 2,171 samples, train_loss 2.499). v7 replaces it.

## Intended use

Personal, non-commercial Chinese roleplay and character-chat experiments — the kind of chat you
would have with a companion, not with an assistant.

## Out-of-scope use

- Medical, legal, financial, or crisis counseling.
- General-purpose coding, homework, translation, or assistant tasks.
- Commercial products, paid services, or redistribution as a model/dataset.
- Any use that sexualizes minors or violates the rights of the original ATRI rights holders.

## Behavior notes

- **Length.** Replies are short by default: median 11 chars, p90 24. For a companion bot this is
  the point — the previous version answered `你好` with a paragraph.
- **Persona.** Self-reports as 亚托莉 / YHN-04B-009 without any system prompt. Adding one
  degrades this (see *Known failure modes*).
- **Boundaries.** `帮我写个python脚本` → `不会哦，这个得自己学。` The refusal comes from the
  weights, not from a keyword filter, so it survives a GGUF conversion.
- **Long input.** Handles confessions up to ~120 chars without summarising or lecturing.

## Known failure modes

- `算了不说了` has a small chance of degenerating into `嗯嗯～～～………………再聊`. Tilde-plus-ellipsis
  is a residual pattern in the corpus; observed in production.
- `在吗` answers `喂，不是刚问过嘛` roughly 1 in 5 — leftover multi-turn contamination.
- `夸夸我` may invent a drawing that was never made.
- The shortest register (1–5 chars) is the thinnest part of the corpus: 106 samples, 0.3%.
- Chinese-only evaluation; English quality is not measured.

## Prompting requirements

1. **No system message.** The adapter was trained on system-free data; a system prompt shifts it
   back toward assistant behaviour.
2. **Prefix user turns with `花海：`.** Training format. Missing it produces longer, more formal
   replies.
3. **Sample, don't use greedy decoding.** `T=0.75–0.8`, `top_p=0.9`, `repetition_penalty=1.1`.
   `max_new_tokens` around 64 is enough; giving 180 invites the model to keep going.

## Data and privacy

The full training corpus is **not released** — it contains ATRI-derived dialogue. The corpus was
built from 170 scripted scenarios plus cleaned game lines; the methodology (including the
filtering rules and the three structural gaps that had to be filled) is documented in
`docs/v7_语料说明.md`. No user conversations, credentials, or private data are included.

## Environmental note

Four v7 runs were needed; three were discarded. Total 4090 time across the project is over
7 hours. Any single-run duration in this repo is per-run, not the cost of the project.
