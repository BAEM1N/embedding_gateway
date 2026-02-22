"""Benchmark: measure embedding latency for every model registered in the gateway."""

import json
import time
import urllib.request
import urllib.error
import sys
import io

# Fix Windows cp949 encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

GATEWAY = "http://localhost:8000"
TEST_INPUT = "Embedding Gateway 벤치마크 테스트 문장입니다. This is a benchmark sentence for measuring embedding latency across different models."


def get_models():
    req = urllib.request.Request(f"{GATEWAY}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data["data"]


def embed(model: str, text: str, timeout: float = 600) -> dict:
    body = json.dumps({"input": text, "model": model}).encode()
    req = urllib.request.Request(
        f"{GATEWAY}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def bench_ollama(models, results):
    """Ollama 모델 벤치마크 (cold = 모델 로드 + 추론, warm = 이미 로드된 상태)."""
    print(f"\n{'='*80}")
    print(f"  Ollama 모델 벤치마크 ({len(models)}개)")
    print(f"  * cold = 모델 로드 + 추론, warm = 이미 로드된 상태에서 추론")
    print(f"{'='*80}")
    for i, m in enumerate(models, 1):
        mid = m["id"]
        sys.stdout.write(f"  [{i:2d}/{len(models)}] {mid:<50s} ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            resp = embed(mid, TEST_INPUT)
            cold = time.perf_counter() - t0
            dims = len(resp["data"][0]["embedding"])
            tokens = resp["usage"]["total_tokens"]

            t0 = time.perf_counter()
            embed(mid, TEST_INPUT)
            warm = time.perf_counter() - t0

            print(f"cold={cold:5.2f}s  warm={warm:5.2f}s  (dim={dims})")
            results.append({
                "model": mid, "backend": "ollama",
                "cold_s": round(cold, 3), "warm_s": round(warm, 3),
                "swap_s": 0, "dims": dims, "tokens": tokens, "status": "ok",
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAIL ({elapsed:.1f}s) - {e}")
            results.append({
                "model": mid, "backend": "ollama",
                "cold_s": round(elapsed, 3), "warm_s": 0, "swap_s": 0,
                "dims": 0, "tokens": 0, "status": f"error: {e}",
            })


def bench_swap_backend(backend_name, models, results):
    """TEI/vLLM 벤치마크 (swap = 컨테이너 재시작 + 모델 로드, infer = 순수 추론)."""
    print(f"\n{'='*80}")
    print(f"  {backend_name.upper()} 모델 벤치마크 ({len(models)}개)")
    print(f"  * swap = 컨테이너 재시작 + 모델 로드, infer = 순수 추론")
    print(f"{'='*80}")
    for i, m in enumerate(models, 1):
        mid = m["id"]
        sys.stdout.write(f"  [{i:2d}/{len(models)}] {mid:<50s} ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            resp = embed(mid, TEST_INPUT)
            total = time.perf_counter() - t0
            dims = len(resp["data"][0]["embedding"])
            tokens = resp["usage"]["total_tokens"]

            t0 = time.perf_counter()
            embed(mid, TEST_INPUT)
            infer = time.perf_counter() - t0

            swap = max(total - infer, 0)
            print(f"swap={swap:6.1f}s  infer={infer:5.3f}s  total={total:6.1f}s  (dim={dims})")
            results.append({
                "model": mid, "backend": backend_name,
                "cold_s": round(total, 3), "warm_s": round(infer, 3),
                "swap_s": round(swap, 3), "dims": dims, "tokens": tokens, "status": "ok",
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAIL ({elapsed:.1f}s) - {e}")
            results.append({
                "model": mid, "backend": backend_name,
                "cold_s": round(elapsed, 3), "warm_s": 0, "swap_s": 0,
                "dims": 0, "tokens": 0, "status": f"error: {e}",
            })


def main():
    models = get_models()
    ollama_models = [m for m in models if m["backend"] == "ollama"]
    tei_models = [m for m in models if m["backend"] == "tei"]
    vllm_models = [m for m in models if m["backend"] == "vllm"]

    results = []

    # ── Ollama ──
    if ollama_models:
        bench_ollama(ollama_models, results)

    # ── TEI ──
    if tei_models:
        bench_swap_backend("tei", tei_models, results)

    # ── vLLM ──
    if vllm_models:
        bench_swap_backend("vllm", vllm_models, results)

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  결과 요약 (warm 기준 정렬 - 순수 추론 속도)")
    print(f"{'='*80}")
    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]

    ok.sort(key=lambda r: r["warm_s"])
    print(f"\n  {'#':>3s}  {'모델':<50s} {'백엔드':<7s} {'warm':>6s} {'cold':>7s} {'swap':>6s} {'dim':>5s}")
    print(f"  {'─'*3}  {'─'*50} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*5}")
    for i, r in enumerate(ok, 1):
        swap_str = f"{r['swap_s']:5.1f}s" if r['swap_s'] > 0 else "    -"
        print(f"  {i:3d}  {r['model']:<50s} {r['backend']:<7s} {r['warm_s']:5.3f}s {r['cold_s']:6.2f}s {swap_str} {r['dims']:5d}")

    if fail:
        print(f"\n  실패: {len(fail)}개")
        for r in fail:
            print(f"    X {r['model']} - {r['status']}")

    print(f"\n  총 {len(ok)}개 성공 / {len(fail)}개 실패")

    # Save JSON
    out_path = "scripts/benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {out_path}\n")


if __name__ == "__main__":
    main()
