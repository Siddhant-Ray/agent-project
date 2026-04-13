#!/usr/bin/env python3
"""
Benchmark uncached 10k/20k/50k/100k vs 500k prefix-cached latency on a running vLLM server.

Examples:
  python bench_vllm_prefix_cache.py --mode 10k
  python bench_vllm_prefix_cache.py --mode 20k
  python bench_vllm_prefix_cache.py --mode 50k
  python bench_vllm_prefix_cache.py --mode 100k
  python bench_vllm_prefix_cache.py --mode 500k

Notes:
- 500k mode:
    * sends the SAME prompt twice
    * ignores latency of the first request
    * reports latency of the second request (cache hit path)
- 10k/20k/50k/100k modes:
    * send one request with a unique nonce to reduce accidental cache reuse
- Prompt text is built from repeated "hi" and trimmed to an exact tokenizer length.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import sys
import time
from typing import Dict, List

import requests
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["10k", "20k", "50k", "75k", "100k", "200k", "500k"],
        required=True,
        help="10k/20k/50k/75k/100k/200k = uncached baseline, 500k = prefix-cached benchmark",
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible server base URL")
    p.add_argument("--model", default="nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
                   help="Served model name")
    p.add_argument("--api-key", default="token-abc123",
                   help="API key configured in vLLM serve")
    p.add_argument("--output-tokens", type=int, default=1,
                   help="max_tokens for generation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=3600.0,
                   help="HTTP timeout in seconds")
    p.add_argument("--tokenizer", default="nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
                   help="Tokenizer used to construct exact-length prompts")
    p.add_argument("--retries", type=int, default=1,
                   help="Extra measured runs in the selected mode")
    return p.parse_args()


def encode_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_exact_token_text(tokenizer, target_tokens: int, base_unit: str = "hi") -> str:
    """
    Build a string whose tokenized length is exactly target_tokens.
    Strategy:
      1) Find a repeated-text length that tokenizes to at least target_tokens.
      2) Encode once, truncate token ids to target_tokens.
      3) Decode back to text.
      4) Verify exact length after re-encoding.
    """
    low = 1
    high = max(8, target_tokens)

    while encode_len(tokenizer, base_unit * high) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        if encode_len(tokenizer, base_unit * mid) < target_tokens:
            low = mid
        else:
            high = mid

    ids = tokenizer(base_unit * high, add_special_tokens=False)["input_ids"][:target_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    final_len = encode_len(tokenizer, text)
    if final_len != target_tokens:
        raise RuntimeError(
            f"Failed to build exact-length prompt: wanted {target_tokens}, got {final_len}"
        )
    return text


def post_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict:
    url = base_url.rstrip("/") + "/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    t1 = time.perf_counter()

    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code}\nResponse: {json.dumps(data, indent=2)[:3000]}"
        )

    usage = data.get("usage", {})
    return {
        "latency_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "text_preview": (data.get("choices", [{}])[0].get("text", "")[:200]),
        "raw": data,
    }


def mode_to_tokens(mode: str) -> int:
    return {
        "10k": 10000,
        "20k": 20000,
        "50k": 50000,
        "75k": 75000,
        "100k": 100000,
        "200k": 200000,
        "500k": 500000,
    }[mode]


def is_uncached_mode(mode: str) -> bool:
    return mode in {"10k", "20k", "50k", "75k", "100k", "200k"}


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    target_tokens = mode_to_tokens(args.mode)

    print(f"Building exact {target_tokens:,}-token prompt...", file=sys.stderr)
    prompt_body = build_exact_token_text(tokenizer, target_tokens)
    verified_len = encode_len(tokenizer, prompt_body)
    print(f"Verified prompt length: {verified_len:,} tokens", file=sys.stderr)

    results: List[float] = []

    if is_uncached_mode(args.mode):
        nonce = secrets.token_hex(16)
        prompt = f"{nonce}\n{prompt_body}"

        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:target_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        final_len = encode_len(tokenizer, prompt)
        if final_len != target_tokens:
            raise RuntimeError(f"Final {args.mode} prompt length mismatch: {final_len}")

        for i in range(args.retries):
            out = post_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.output_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            results.append(out["latency_s"])
            print(json.dumps({
                "mode": f"{args.mode}_uncached",
                "run": i + 1,
                "latency_s": out["latency_s"],
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "total_tokens": out["total_tokens"],
            }, indent=2))

    else:
        # 500k cached path: send same prompt twice, measure the second.
        prompt = prompt_body

        print("Warm request (fills prefix cache)...", file=sys.stderr)
        warm = post_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.output_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        print(json.dumps({
            "mode": "500k_cached",
            "phase": "warm",
            "latency_s": warm["latency_s"],
            "prompt_tokens": warm["prompt_tokens"],
            "completion_tokens": warm["completion_tokens"],
            "total_tokens": warm["total_tokens"],
        }, indent=2))

        for i in range(args.retries):
            measured = post_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.output_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            results.append(measured["latency_s"])
            print(json.dumps({
                "mode": "500k_cached",
                "phase": "measured_second_request",
                "run": i + 1,
                "latency_s": measured["latency_s"],
                "prompt_tokens": measured["prompt_tokens"],
                "completion_tokens": measured["completion_tokens"],
                "total_tokens": measured["total_tokens"],
            }, indent=2))

    if results:
        summary = {
            "mode": args.mode,
            "runs": len(results),
            "mean_latency_s": statistics.mean(results),
            "median_latency_s": statistics.median(results),
            "min_latency_s": min(results),
            "max_latency_s": max(results),
            "output_tokens": args.output_tokens,
        }
        print(json.dumps({"summary": summary}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())