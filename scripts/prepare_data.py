"""把「你自己的语料」编译成训练格式（messages jsonl）。

支持的输入（放进 data/corpus/ 或任意目录，脚本会递归扫描）：

1. **turns jsonl**（推荐，最好写）
   {"id":"x1","scene":"傍晚的海边","user_role":"花海","turns":[["u","..."],["a","..."]]}

2. **OpenAI messages jsonl**
   {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

3. **ShareGPT jsonl**
   {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}

4. **纯文本剧本 .txt / .md**
   自动识别这几种行格式：
       亚托莉「我是高性能的嘛！」
       亚托莉：我是高性能的嘛！
       [亚托莉] 我是高性能的嘛！
   脚本会把「亚托莉说的话」当 assistant，其他角色/旁白合并成 user，
   按上下文滑窗切成多条训练样本。

用法：
    python scripts/prepare_data.py                          # 扫 data/corpus
    python scripts/prepare_data.py --input D:\我的语料 --stats-only
    python scripts/prepare_data.py --input data/corpus --valid-ratio 0.05

输出：
    data/processed/train.jsonl
    data/processed/valid.jsonl
    data/processed/report.json     # 统计 + 被丢弃的原因
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_messages, build_system_prompt, project_root  # noqa: E402

ATRI_NAMES = {"亚托莉", "アトリ", "ATRI", "atri", "萝卜子", "亞托莉", "亚托莉(ATRI)"}

BANNED = ["<think>", "</think>", "作为AI", "作为 AI", "语言模型", "AI助手", "AI 助手",
          "人工智能助手", "ChatGPT"]

# 剧本行：名字「台词」 / 名字：台词 / [名字] 台词
RE_QUOTE = re.compile(r"^\s*(?P<name>[^\s：:「」『』\[\]（）()]{1,16})\s*[「『](?P<text>.+?)[」』]\s*$")
RE_COLON = re.compile(r"^\s*(?P<name>[^\s：:]{1,16})\s*[：:]\s*(?P<text>.+?)\s*$")
RE_BRACKET = re.compile(r"^\s*\[(?P<name>[^\]]{1,16})\]\s*(?P<text>.+?)\s*$")


def norm(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?~～…「」『』（）()]", "", s)


# ------------------------------------------------------------------ 各种 loader
def load_turns(rec: dict) -> dict | None:
    turns = rec.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    out = []
    for t in turns:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            role = "u" if str(t[0]).lower() in ("u", "user", "human", "用户") else "a"
            out.append((role, str(t[1]).strip()))
        elif isinstance(t, dict):
            role = "a" if str(t.get("role", "")).lower() in ("a", "assistant", "gpt") else "u"
            out.append((role, str(t.get("content") or t.get("text") or "").strip()))
    return {"id": rec.get("id", ""), "scene": rec.get("scene"), "user_role": rec.get("user_role"),
            "turns": out, "system": rec.get("system"), "group": rec.get("group")}


def load_messages(rec: dict) -> dict | None:
    msgs = rec.get("messages")
    if not isinstance(msgs, list):
        return None
    system, turns = None, []
    for m in msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        elif role == "user":
            turns.append(("u", content))
        elif role == "assistant":
            turns.append(("a", content))
    return {"id": rec.get("id", ""), "scene": rec.get("scene"), "user_role": rec.get("user_role"),
            "turns": turns, "system": system, "group": rec.get("group")}


def load_sharegpt(rec: dict) -> dict | None:
    convs = rec.get("conversations") or rec.get("conversation")
    if not isinstance(convs, list):
        return None
    turns = []
    for m in convs:
        frm = str(m.get("from", "")).lower()
        val = str(m.get("value") or "").strip()
        if not val:
            continue
        if frm in ("human", "user"):
            turns.append(("u", val))
        elif frm in ("gpt", "assistant", "ai"):
            turns.append(("a", val))
    return {"id": rec.get("id", ""), "scene": rec.get("scene"), "user_role": rec.get("user_role"),
            "turns": turns, "system": None, "group": rec.get("group")}


def parse_script_text(text: str, src_name: str, max_ctx: int = 6) -> list[dict]:
    """把纯文本剧本切成对话样本：目标是亚托莉的每一句。"""
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = RE_QUOTE.match(s) or RE_BRACKET.match(s) or RE_COLON.match(s)
        if m:
            lines.append((m.group("name").strip(), m.group("text").strip()))
        else:
            lines.append(("", s))   # 旁白

    samples, buf = [], []
    for name, txt in lines:
        if not txt:
            continue
        is_atri = name in ATRI_NAMES
        if is_atri:
            if buf:
                turns = buf[-max_ctx:] + [("a", txt)]
                # 保证 u/a 交替、以 a 结尾
                merged, last_role = [], None
                for r, t in turns:
                    if last_role == r:
                        merged[-1] = (r, merged[-1][1] + "\n" + t)
                    else:
                        merged.append((r, t))
                        last_role = r
                if merged[0][0] == "a":
                    merged = merged[1:]
                if len(merged) >= 3 and merged[-1][0] == "a":
                    samples.append({"id": "", "scene": None, "user_role": None,
                                    "turns": merged, "system": None})
            buf = []
        else:
            buf.append(("u", (f"{name}：{txt}" if name else txt)))
            if len(buf) > max_ctx:
                buf = buf[-max_ctx:]
    print(f"  [剧本] {src_name}: 识别 {len(lines)} 行 → {len(samples)} 条样本")
    return samples


# ------------------------------------------------------------------ 校验
def validate(rec: dict, min_asst_chars: int, max_asst_chars: int) -> list[str]:
    errs = []
    turns = rec.get("turns") or []
    if len(turns) < 2:
        return ["轮数<2"]
    if turns[0][0] != "u" or turns[-1][0] != "a":
        return ["未以 u 开头 / a 结尾"]
    for role, text in turns:
        if not text or not text.strip():
            errs.append("空台词")
        if any(b in text for b in BANNED):
            errs.append("含禁用词")
        if role == "a":
            if len(text) < min_asst_chars:
                errs.append(f"assistant 过短(<{min_asst_chars})")
            if len(text) > max_asst_chars:
                errs.append(f"assistant 过长(>{max_asst_chars})")
    return errs


# ------------------------------------------------------------------ 主流程
def collect(input_dir: str) -> tuple[list[dict], Counter]:
    records, drop = [], Counter()
    files = [p for p in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
             if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".jsonl", ".json", ".txt", ".md")]
    if not files:
        print(f"[!] {input_dir} 下没有 .jsonl/.json/.txt/.md 文件")
    for fp in sorted(files):
        ext = os.path.splitext(fp)[1].lower()
        print(f"[读取] {os.path.relpath(fp, input_dir)}")
        if ext in (".txt", ".md"):
            with open(fp, encoding="utf-8", errors="ignore") as f:
                records += parse_script_text(f.read(), os.path.basename(fp))
            continue
        if ext == ".json":
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
        else:
            items = []
            with open(fp, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        drop["JSON 解析失败"] += 1
        for it in items:
            if not isinstance(it, dict):
                drop["非对象"] += 1
                continue
            rec = load_turns(it) or load_messages(it) or load_sharegpt(it)
            if rec is None:
                drop["无法识别的结构"] += 1
                continue
            records.append(rec)
    return records, drop


def to_record(d: dict, rng: random.Random) -> dict:
    scene, role = d.get("scene"), d.get("user_role")
    if d.get("system"):
        system = d["system"]
    else:
        variant = rng.choices(["full", "role", "none"], weights=[0.6, 0.2, 0.2])[0]
        system = build_system_prompt(scene if variant == "full" else None,
                                     role if variant in ("full", "role") else None)
    msgs = build_messages(d["turns"], system)
    return {"id": d.get("id", ""), "messages": msgs,
            "meta": {"scene": scene, "user_role": role,
                     "n_turns": len(d["turns"]),
                     "group": d.get("group"),
                     "asst_chars": sum(len(t) for r, t in d["turns"] if r == "a")}}


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(root, "data", "corpus"))
    ap.add_argument("--out-dir", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--group-by", default="", help="按记录里的字段做分组切分，避免同一场景的变体泄漏到 valid")
    ap.add_argument("--min-asst-chars", type=int, default=4)
    ap.add_argument("--max-asst-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    records, drop = collect(args.input)

    # 校验 + 去重
    kept, seen = [], set()
    for r in records:
        errs = validate(r, args.min_asst_chars, args.max_asst_chars)
        if errs:
            drop[errs[0]] += 1
            continue
        key = norm("".join(t for _, t in r["turns"]))
        if key in seen:
            drop["重复"] += 1
            continue
        seen.add(key)
        kept.append(r)

    n_turns = sum(len(r["turns"]) for r in kept)
    asst_chars = sum(len(t) for r in kept for role, t in r["turns"] if role == "a")
    print("\n=== 统计 ===")
    print(f"原始样本 {len(records)} → 可用 {len(kept)}")
    print(f"总轮数 {n_turns}，assistant 字数 {asst_chars}")
    if drop:
        print("丢弃原因：")
        for k, v in drop.most_common():
            print(f"  {k}: {v}")
    if not kept:
        print("\n[!] 没有可用样本。把语料放进 data/corpus/ 后再跑。")
        return
    if args.stats_only:
        return

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.group_by:
        groups: dict[str, list[dict]] = {}
        for r in kept:
            g = r.get(args.group_by) or r.get("id") or norm("".join(t for _, t in r["turns"]))[:80]
            groups.setdefault(str(g), []).append(r)
        keys = list(groups)
        rng.shuffle(keys)
        target_valid = max(1, int(len(kept) * args.valid_ratio)) if len(kept) > 20 else 0
        valid_keys, n_valid = [], 0
        for k in keys:
            if target_valid and n_valid >= target_valid and valid_keys:
                break
            valid_keys.append(k)
            n_valid += len(groups[k])
        vset = set(valid_keys)
        train = [r for k in keys if k not in vset for r in groups[k]]
        valid = [r for k in keys if k in vset for r in groups[k]]
        print(f"[分组切分] {len(groups)} 组 → train {len(train)} / valid {len(valid)}")
    else:
        n_valid = max(1, int(len(kept) * args.valid_ratio)) if len(kept) > 20 else 0
        train, valid = kept[n_valid:], kept[:n_valid]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, subset in (("train", train), ("valid", valid)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for d in subset:
                f.write(json.dumps(to_record(d, rng), ensure_ascii=False) + "\n")
        print(f"[写出] {path}  {len(subset)} 条")

    report = {
        "input": os.path.abspath(args.input),
        "raw_samples": len(records),
        "kept_samples": len(kept),
        "train_samples": len(train),
        "valid_samples": len(valid),
        "turns": n_turns,
        "assistant_chars": asst_chars,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "group_by": args.group_by or None,
        "min_asst_chars": args.min_asst_chars,
        "max_asst_chars": args.max_asst_chars,
        "dropped": dict(drop),
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[写出] {report_path}")

    # 留 3 条给人眼看
    print("\n=== 抽样（前 2 条）===")
    for d in kept[:2]:
        for role, t in d["turns"][:6]:
            print(f"  {'用户' if role == 'u' else '亚托莉'}: {t[:60]}")
        print("  ---")


if __name__ == "__main__":
    main()
