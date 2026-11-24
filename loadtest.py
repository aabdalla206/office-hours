"""Simulate a class asking questions at once, against the vLLM server directly.

Measures time-to-first-token (TTFT), end-to-end latency (p50/p95), output tokens/sec, and
cost per 1,000 questions from the GPU's hourly price. Run it once with vLLM started with
--enable-prefix-caching and once with --no-enable-prefix-caching to see what caching buys.

    python loadtest.py --students 50 --questions questions.txt --gpu-cost-per-hour 0.35 --label cache-on

GPU price is YOUR input (Colab, Lambda, Modal, a lab machine). For an API comparison, pass the
provider's current per-token prices with --api-in-per-m and --api-out-per-m; nothing is hardcoded.
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

import guardrail
from prompting import PromptBuilder


async def one(client, url, model, messages, max_tokens):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
               "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, usage = None, {}
    async with client.stream("POST", url, json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            msg = json.loads(line[6:])
            if msg.get("usage"):
                usage = msg["usage"]
            if ttft is None and msg.get("choices") and msg["choices"][0].get("delta", {}).get("content"):
                ttft = time.perf_counter() - t0
    return {"ttft": ttft or 0.0, "latency": time.perf_counter() - t0,
            "in_tok": usage.get("prompt_tokens", 0), "out_tok": usage.get("completion_tokens", 0)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--questions", default="questions.txt")
    ap.add_argument("--students", type=int, default=50, help="concurrent requests")
    ap.add_argument("--total", type=int, default=200, help="total questions to send")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--gpu-cost-per-hour", type=float, required=True)
    ap.add_argument("--api-in-per-m", type=float, help="API $ per 1M input tokens (optional)")
    ap.add_argument("--api-out-per-m", type=float, help="API $ per 1M output tokens (optional)")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    builder = PromptBuilder()
    qs = [q.strip() for q in open(args.questions) if q.strip()]
    qs = [q for q in qs if guardrail.check(q) is None]  # refused requests never reach the GPU
    prompts = [builder.messages(random.choice(qs)) for _ in range(args.total)]

    sem = asyncio.Semaphore(args.students)
    async with httpx.AsyncClient(timeout=None) as client:
        await one(client, args.url, args.model, prompts[0], 1)  # warmup

        async def guarded(p):
            async with sem:
                return await one(client, args.url, args.model, p, args.max_tokens)

        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0

    out_tok = sum(r["out_tok"] for r in res)
    in_tok = sum(r["in_tok"] for r in res)
    cost_per_1k = args.gpu_cost_per_hour * (wall / 3600) / len(res) * 1000
    summary = {
        "label": args.label, "students": args.students, "questions": len(res),
        "avg_prompt_tokens": round(in_tok / len(res)),
        "ttft_p50_s": round(pct([r["ttft"] for r in res], 50), 3),
        "ttft_p95_s": round(pct([r["ttft"] for r in res], 95), 3),
        "latency_p50_s": round(statistics.median(r["latency"] for r in res), 2),
        "latency_p95_s": round(pct([r["latency"] for r in res], 95), 2),
        "output_tok_per_s": round(out_tok / wall, 1),
        "questions_per_min": round(len(res) / wall * 60, 1),
        "gpu_cost_per_1k_questions_usd": round(cost_per_1k, 4),
    }
    if args.api_in_per_m is not None and args.api_out_per_m is not None:
        api = (in_tok * args.api_in_per_m + out_tok * args.api_out_per_m) / 1e6
        summary["api_cost_per_1k_questions_usd"] = round(api / len(res) * 1000, 4)
    print(json.dumps(summary, indent=2))
    with open("loadtest_results.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
