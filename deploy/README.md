# 部署：llama.cpp + Q8_0 + v7 LoRA

## 为什么不用 ollama

两个硬原因：

1. **v6 时代的接入配置里有 `dry_multiplier` / `dry_base`** —— 这是 llama.cpp 的 DRY 采样
   参数，ollama 不认。要继续用 DRY 就得上 llama.cpp。
2. llama.cpp 支持**运行时挂载 LoRA**（`--lora`），不用把 323MB adapter 合并进 7.6GB 底模。
   换 adapter 只是换一个文件，改 LoRA 强度也只是加 `--lora-scaled`。

## 步骤

### 1. 准备 Q8_0 底模

必须是**纯 `Qwen2.5-7B-Instruct` 转出来的**，不能是已经合并过旧 LoRA 的权重，否则叠加
会出错。

```bash
python convert_hf_to_gguf.py models/Qwen2.5-7B-Instruct --outtype q8_0 \
    --outfile atri-deploy/q8/Qwen2.5-7B-Instruct-Q8_0.gguf
```

### 2. 把 v7 LoRA 转成 gguf

LoRA 的 delta 是低秩小量，再量化会放大误差（q8_0 转出来只有 86MB，和 f16 差 74MB，
没必要省）。**用 f16**：

```bash
python convert_lora_to_gguf.py --base models/Qwen2.5-7B-Instruct \
    --outtype f16 --outfile atri-deploy/q8/atri-v7-lora.gguf \
    models/atri-v7-lora
```

`--base` 只读 config.json 和 tokenizer，不用把 15GB 权重放在本地。

### 3. 放好文件

```
atri-deploy/q8/
├── Qwen2.5-7B-Instruct-Q8_0.gguf   7.6G   底模
├── atri-v7-lora.gguf               155M   v7 LoRA
├── chat-template.jinja                     ← 本目录
└── start.sh                                ← 本目录
```

### 4. 起服务

```bash
chmod +x start.sh
./start.sh
# 或者交给 systemd：
sudo cp atri-llama.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now atri-llama
```

### 5. 测

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"atri","messages":[{"role":"user","content":"我好累"}],
       "max_tokens":64,"temperature":0.75,"top_p":0.9,"repeat_penalty":1.1}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## 模板为什么长这样

`chat-template.jinja` 做了三件 v7 特有的事，缺一个都会退化：

1. **丢掉所有 system 消息**（模板里没有 system 分支）。v7 的语料 100% 不带 system，
   人格在权重里。上游（AstrBot / OpenAI SDK）硬塞一条 system 进来，模板直接无视它。
2. **user 内容前补「花海：」**。训练时每条 user 都是这个格式，不补前缀就是训练/推理分布
   错位 —— 实测表现为回复变长、语气变客套。
3. **保留全部历史**。旧模板只取最后一条 user，把历史全扔了，等于每轮都从零开始。

## 实测（Ryzen 5 4600U，6C12T，DDR4 双通道，纯 CPU）

| 项目 | 数值 |
|---|---|
| 生成速度 | **2.2 tok/s**（20 字回复约 9 秒） |
| 模型加载 | 5.3 s（mmap，权重走 page cache，实际 RSS 只占 3.7G） |
| prefill 冷启动（347 token） | **16.5 s** |
| prefill 命中前缀缓存（15 token） | **1.0 s** |

**瓶颈是 prefill 不是生成。** llama-server 会复用相同前缀的 KV cache，所以连续对话每轮
只算新增的二十来个 token，体感正常。但前提是上游别往 prompt 里塞变化内容（当前时间、
随机昵称、每轮不同的 RAG 片段）—— 一塞缓存全废，每条消息都要等十几秒。

## 接 AstrBot

服务提供商 → OpenAI 兼容：

| 字段 | 值 |
|---|---|
| API Base | `http://127.0.0.1:8088/v1` |
| Key | 任意（服务不校验） |
| 模型 | `atri` |
| 上下文上限 | 16384（和 `-c` 一致） |
| system prompt | **留空** |

AstrBot 跑在 docker 里的话，`127.0.0.1` 要换成 `172.17.0.1` 或宿主机的 docker 网桥 IP。

## 已知问题

- `--parallel 1` 只有一个 slot，群聊多人同时说话会排队。
- 「算了不说了」这类收尾话有概率退化成「嗯嗯～～～………………再聊」——波浪号加省略号
  是 v7 语料的残留模式，下版清。
- 纯 CPU 想提速，最划算的是 q8 换 q6_k：生成快约 40%、内存省 1.5G，6bit 基本无感。
