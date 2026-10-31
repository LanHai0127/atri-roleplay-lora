# Model Card: Atri v6.3 LoRA

## Model details

- **Base model:** Qwen2.5-7B-Instruct (Apache-2.0)
- **Adapter:** LoRA, rank 16, trained with alpha 32, released with `lora_alpha=24` (0.75×)
- **Task:** Chinese character roleplay as 亚托莉 (ATRI, YHN-04B-009)
- **Training:** QLoRA 4bit NF4, 2 epochs, 2,171 train / 154 valid ChatML samples
- **Hardware:** 1 × RTX 4090 24 GB, 33 minutes
- **Final metrics:** train_loss 2.499, eval_loss 2.133, peak VRAM 9.86 GB

## Intended use

Personal, non-commercial Chinese roleplay and character-chat experiments.

## Out-of-scope use

- Medical, legal, financial, or crisis counseling.
- General-purpose coding, homework, translation, or assistant tasks.
- Commercial products, paid services, or redistribution as a model/dataset.
- Any use that sexualizes minors or violates the rights of the original ATRI rights holders.

## Behavior notes

v6.3 adds:

- Refusal of code / homework / papers / PPT / tables / translation / web-search requests.
- “这是什么？我不懂。” for specialized concepts outside the character’s knowledge.
- Everyday knowledge (tides, simple arithmetic) is still answered.
- A keyword-based self-harm safety guard in `scripts/common.py`.

The guard is not part of the adapter and is **not** active in a bare GGUF/llama.cpp deployment.

## Limitations

- Some preference questions are evasive.
- Occasional “不会离开” over-promising conflicts with the “can’t promise forever” setting.
- Name drift can still occur; `clean_reply()` repairs the common cases.
- Quantization to Q4/Q3 may weaken the boundary behavior.
- Chinese-only evaluation; English quality is not measured.

## Data and privacy

The full training corpus is **not released** because it contains ATRI-derived material.
No user conversations, credentials, or private data are included.
