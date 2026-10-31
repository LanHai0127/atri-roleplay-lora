"""多线程断点续传下载 HF 模型（hf-mirror 单连接常被限速，这里分块并发）。

用法：
    python scripts/fetch_model.py ReXeeD/Luminus-1.5B-Roleplay models/Luminus-1.5B-Roleplay
    python scripts/fetch_model.py <repo> <dir> --threads 8 --file model.safetensors
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dsh-fetch/1.0"}


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
                cl = r.headers.get("Content-Length")
                return int(cl) if cl else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_chunk(url: str, start: int, end: int, path: str, retries: int = 12) -> int:
    """下载 [start, end] 到 path，返回字节数。断点续传：path 已有的部分跳过。"""
    done = os.path.getsize(path) if os.path.exists(path) else 0
    total = end - start + 1
    if done >= total:
        return total
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start + done}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
            if done >= total:
                return done
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 * (attempt + 1)))
            done = os.path.getsize(path) if os.path.exists(path) else 0
    return done


def download_file(repo: str, filename: str, dest_dir: str, threads: int = 6, chunk_mb: int = 24):
    url = f"{ENDPOINT}/{repo}/resolve/main/{filename}"
    final = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)

    size = head_size(url)
    if not size:
        sys.exit(f"无法获取 {filename} 大小")
    if os.path.exists(final) and os.path.getsize(final) == size:
        print(f"[跳过] {filename} 已完整 ({size/2**20:.1f} MB)")
        return final
    print(f"[下载] {filename}  {size/2**20:.1f} MB  {threads} 线程")

    chunk = chunk_mb * 2**20
    parts = []
    tmp_dir = os.path.join(dest_dir, ".parts")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, start in enumerate(range(0, size, chunk)):
        end = min(start + chunk - 1, size - 1)
        parts.append((i, start, end, os.path.join(tmp_dir, f"{filename}.p{i:05d}")))
    for p in parts:
        if os.path.exists(p[3]) and os.path.getsize(p[3]) > (p[2] - p[1] + 1):
            os.remove(p[3])

    t0 = time.time()
    last = [0]

    def report():
        got = sum(os.path.getsize(p[3]) for p in parts if os.path.exists(p[3]))
        el = time.time() - t0
        print(f"  {got/2**20:7.1f}/{size/2**20:.1f} MB  "
              f"{got/2**20/max(el,1e-9):.2f} MB/s  {el:.0f}s", flush=True)

    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(fetch_chunk, url, s, e, p): p for _, s, e, p in parts}
        while futs:
            done, pending = cf.wait(futs, timeout=20)
            for f in done:
                p = futs.pop(f)
                if f.exception():
                    sys.exit(f"分块失败 {p}: {f.exception()}")
            report()

    print("[合并] ...")
    with open(final, "wb") as out:
        for _, _, _, p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(8 << 20)
                    if not b:
                        break
                    out.write(b)
    for _, _, _, p in parts:
        os.remove(p)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    got = os.path.getsize(final)
    print(f"[完成] {final}  {got/2**20:.1f} MB  {'OK' if got == size else '大小不符!'}")
    return final


def list_repo_files(repo: str) -> list[str]:
    """通过 hf-mirror 的 API 拿仓库文件列表；失败则退回常见文件名。"""
    import json
    url = f"{ENDPOINT}/api/models/{repo}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return [s["rfilename"] for s in data.get("siblings", [])]
    except Exception as e:
        print(f"[warn] 无法列出仓库文件（{e}），使用默认列表")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="下载仓库全部文件（跳过 .md/.gitattributes）")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--chunk-mb", type=int, default=24)
    args = ap.parse_args()

    if args.files:
        files = args.files
    elif args.all:
        files = [
            f for f in list_repo_files(args.repo)
            if not f.endswith((".md", ".gitattributes", ".pth", ".bin", ".msgpack", ".h5"))
            and not f.startswith(".")
        ]
        if not files:
            files = ["model.safetensors"]
        print(f"[列表] {len(files)} 个文件：{files}")
    else:
        files = ["model.safetensors"]

    # 大文件优先，保证多个 safetensors 分片尽快并行下载
    files.sort(key=lambda f: 0 if f.endswith(".safetensors") else 1)
    for fn in files:
        try:
            download_file(args.repo, fn, args.dest, args.threads, args.chunk_mb)
        except SystemExit as e:
            print(f"[跳过] {fn}: {e}")
        except Exception as e:
            print(f"[跳过] {fn}: {e}")



if __name__ == "__main__":
    main()
