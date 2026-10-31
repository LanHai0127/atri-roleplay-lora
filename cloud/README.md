# 云 GPU 训练包 · 亚托莉 7B

这个目录是给 **24 GB 显卡**（RTX 4090 / A10 / L4 / A5000 等）准备的 7B QLoRA 训练包。  
本机 8 GB 已经实测跑不动 7B（384 长度峰值 9.11 GB，256 也有 8.87 GB），所以 7B 必须上云。

> 包里只放 **v5 基线数据**（train 2,876 / valid 151）。  
> v6 外部队列 370 条还是“咨询师口吻候选”，没改写成亚托莉，**没有放进来**；先跑 v5 基线，再决定要不要加 v6。

---

## 1. 机器要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| GPU | 24 GB（4090 / A10 / L4 / A5000） | **RTX 5090 32 GB（一次全量）** |
| 内存 | 32 GB | 64 GB |
| 磁盘 | 50 GB | 100 GB（要 merge + GGUF 时） |
| 系统 | Ubuntu 20.04/22.04 | Ubuntu 22.04 |
| CUDA | 12.1+ | 12.4 |
| Python | 3.10+ | 3.11 |

AutoDL 选 **PyTorch 2.4+ / CUDA 12.1 / Python 3.10** 的镜像即可，不要自己重装 torch。

---

## 2. 上传与解压

把 `atri_cloud_bundle.zip` 传到云实例（AutoDL 的 JupyterLab 直接拖拽，或 `scp`）：

```bash
cd /root/autodl-tmp
unzip atri_cloud_bundle.zip -d atri
cd atri
```

目录应该是：

```text
atri/
├─ scripts/
│  ├─ train_lora.py
│  ├─ merge_lora.py
│  ├─ eval.py
│  ├─ chat.py
│  └─ common.py
├─ data/processed_v5/
│  ├─ train.jsonl
│  └─ valid.jsonl
├─ data/eval/
│  ├─ atri-v5-prompts.jsonl
│  └─ atri-v6-companion-prompts.jsonl
└─ cloud/
   ├─ setup.sh
   ├─ download_model.sh
   ├─ train_7b.sh
   ├─ train_7b_5090.sh
   └─ merge_and_export.sh
```

---

## 3. 安装依赖

```bash
bash cloud/setup.sh
```

它只装 `transformers / peft / bitsandbytes / accelerate / datasets / sentencepiece`，**不会动镜像里的 torch**。

如果 `bitsandbytes` 报 CUDA 版本不匹配，先看：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

AutoDL 的 PyTorch 2.4 + CUDA 12.1 镜像一般直接可用。

---

## 4. 下载基座

```bash
bash cloud/download_model.sh
```

默认下到 `models/Qwen2.5-7B-Instruct`，约 15 GB。优先走 ModelScope；失败会回退到 `hf-mirror.com`。

也可以手动：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir models/Qwen2.5-7B-Instruct
```

---

## 5. 训练 v5 基线

```bash
bash cloud/train_7b.sh
```

默认参数：

```text
max_len 1024（v5 最大 452，不会截断）
1 epoch，lr 5e-5，batch 1，grad-accum 8
QLoRA 4bit，r=8，alpha=16，dropout=0.1
save_steps=50，只存 adapter（约 80 MB）
输出 outputs/atri-v5-7b-cloud/
```

4090 上大约 **30–60 分钟**（含 eval），显存预计 10–14 GB。  
日志在 `logs/cloud-train-*.log`。

> 第一次先看 `checkpoint-50` 的 loss；如果 loss 正常下降，再等全量。  
> 训练中途断了也没关系：`save_only_model=True`，拿最近的 `checkpoint-*` 继续或直接评测。

---

## 5.5 RTX 5090 32 GB：一次全量跑完

5090 的 32 GB 足够把 v5 的 2,876 条全部跑完，不跳任何样本：

```bash
bash cloud/train_7b_5090.sh
```

它和 5 的区别只有三处：

```text
max_len 1024（v5 最大 452，全部保留）
batch 2，grad-accum 4（等效 batch 8，和原来一致）
输出 outputs/atri-v5-7b-5090-full/
```

预计 **15–30 分钟**（含 151 条 valid 的 eval），显存预计 14–20 GB。日志在 `logs/cloud-train-5090-*.log`。

**5090 的两个硬条件：**

1. `torch` 必须是 **CUDA 12.8+** 的 wheel（5090 是 sm_120）。先用：
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
   ```
   期望看到 `12.8` 和 `(12, 0)`；如果 CUDA 是 12.1，会报 `no kernel image is available`。
2. `bitsandbytes >= 0.50`（脚本的 `requirements.txt` 已经固定）。旧版 bnb 在 sm_120 上会直接失败。

如果云镜像的 torch 是 cu121，不要在这个镜像里硬装 5090；换一个 **PyTorch 2.8+ / CUDA 12.8** 的镜像，或者按官方命令装 cu128 wheel。

---

## 5.6 如果是本机 Windows 5090

不要用 `cloud/*.sh`，在项目根目录直接跑：

```powershell
.\env\Scripts\python.exe scripts\train_lora.py `
  --model .\models\Qwen2.5-7B-Instruct `
  --train .\data\processed_v5\train.jsonl `
  --valid .\data\processed_v5\valid.jsonl `
  --max-len 1024 --epochs 1 --lr 5e-5 `
  --batch 2 --grad-accum 4 `
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 `
  --load-4bit on `
  --out .\outputs\atri-v5-7b-5090-full
```

跑之前关掉 QQ / 浏览器 / ComfyUI；5090 32 GB 不会像 8 GB 那样踩内存，但 Windows 的 WDDM 仍建议留 2–3 GB 余量。预计 20–35 分钟。

---

## 5.7 v6-clean：先清洗，再谈加数据

v5 的 4090 基线（1 epoch / lr 5e-5 / r=8）跑完后，人设和指令跟随都出了问题：会出现“主人”“陪睡”“嗯哼”，以及“叫我一声→叫吧”这种答非所问。原因不是 epoch 不够，而是 v5 里模板句重复率太高、原作擦边称呼没清。

v6-clean 已生成在 `data/processed_v6/`：

```bash
TRAIN=data/processed_v6/train.jsonl \
VALID=data/processed_v6/valid.jsonl \
OUT=outputs/atri-v6-7b-clean \
EPOCHS=2 LR=2e-5 LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.1 \
bash cloud/train_7b.sh
```

预计 4090 24 GB 约 30 分钟；峰值显存约 10.5 GB。先看这版人设是否止住，再加外部 370 条。详见 `docs/v6_语料说明.md`。

---

## 6. 取回 adapter（推荐，最省事）

训练完只需要下载：

```text
outputs/atri-v5-7b-cloud/adapter_config.json
outputs/atri-v5-7b-cloud/adapter_model.safetensors
outputs/atri-v5-7b-cloud/tokenizer.json
outputs/atri-v5-7b-cloud/tokenizer_config.json
outputs/atri-v5-7b-cloud/chat_template.jinja
```

约 80 MB。**5090 全量跑的把路径换成 `outputs/atri-v5-7b-5090-full`。** 拿回本机后：

```powershell
.\env\Scripts\python.exe scripts\merge_lora.py `
  --base .\models\Qwen2.5-7B-Instruct `
  --adapter .\outputs\atri-v5-7b-cloud `
  --out .\outputs\atri-v5-7b-merged
```

本机 31 GB 内存可以 merge 7B bf16（约 15 GB 权重）；merge 脚本已修 PEFT 0.21 的 `merge_and_unload` 问题，3B 实测通过。

---

## 6.5 可选：先在云上评测，再决定要不要下载

云上有 24 GB，跑 20+20 条 prompt 很快：

```bash
python scripts/eval.py \
  --base models/Qwen2.5-7B-Instruct \
  --adapter outputs/atri-v5-7b-cloud \
  --prompts data/eval/atri-v5-prompts.jsonl \
  --out-dir eval/cloud-v5

python scripts/eval.py \
  --base models/Qwen2.5-7B-Instruct \
  --adapter outputs/atri-v5-7b-cloud \
  --prompts data/eval/atri-v6-companion-prompts.jsonl \
  --out-dir eval/cloud-v6
```

输出是 markdown，基座和微调后并排。先看这 40 条，再决定是下载 adapter 还是继续调数据；不要只看 eval_loss。

---

## 7. 可选：云端 merge + GGUF

只有磁盘 ≥100 GB 再做：

```bash
bash cloud/merge_and_export.sh
```

它会：

1. merge 成 bf16 完整模型（约 15 GB）；
2. 转 f16 GGUF（约 15 GB）；
3. 量化成 `Q4_K_M`（约 4.5 GB）；
4. 删掉中间 f16，保留：

```text
outputs/atri-v5-7b-merged/
exports/atri-7b-Q4_K_M.gguf
```

磁盘不够就设 `SKIP_GGUF=1`，只 merge：

```bash
SKIP_GGUF=1 bash cloud/merge_and_export.sh
```

llama.cpp 默认从 GitHub clone；如果云上 GitHub 慢：

```bash
LLAMA_REPO=https://ghproxy.net/https://github.com/ggerganov/llama.cpp.git \
  bash cloud/merge_and_export.sh
```

---

## 8. 费用与时长参考

| 项目 | 参考 |
|---|---|
| 4090 24 GB | 约 ¥2/小时（AutoDL 等） |
| v5 1 epoch | 30–60 分钟 |
| 总费用 | 约 ¥2–5 |
| adapter | 约 80 MB |
| 部署 | 仍是本机 GGUF Q4_K_M + llama.cpp |

训练完记得 **关机/释放实例**，别只关网页。

---

## 9. 常见问题

**Q: 为什么不用 512/384？**  
A: 云上 24 GB 没必要压长度；v5 最大 452，`max_len 1024` 不会截断任何样本。

**Q: 能不能直接训 v6 外部数据？**  
A: 现在不能。370 条只是候选，assistant 还是“咨询师”口吻；直接训会把亚托莉变成心理咨询师。先跑 v5 基线，再按 `docs/外部队列调研_v6.md` 改写。

**Q: 云上要不要装 DeepSpeed / FlashAttention？**  
A: 不需要。这个脚本是单卡 QLoRA，`sdpa` + gradient checkpointing 就够；装 FlashAttention 反而容易和镜像 CUDA 打架。

**Q: 训练完 adapter 怎么验证？**  
A: 本机 `chat.py --adapter ...`，再用 `data/eval/atri-v5-prompts.jsonl` 和 `data/eval/atri-v6-companion-prompts.jsonl` 各跑 20 条。

**Q: 数据会不会泄露？**  
A: 用私有实例；训练完删除实例和数据。不要把 `data/` 或 adapter 传到公共仓库。
