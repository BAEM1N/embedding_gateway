#!/bin/bash
# Ollama 임베딩 모델 전체 다운로드 스크립트
set -euo pipefail

MODELS=(
    # Qwen3 Embedding (기본 양자화)
    "qwen3-embedding:0.6b"
    "qwen3-embedding:4b"
    "qwen3-embedding:8b"

    # Qwen3 Embedding F16 (HuggingFace GGUF, GPU+CPU 하이브리드)
    "hf.co/Qwen/Qwen3-Embedding-4B-GGUF:F16"
    "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16"

    # Google
    "embeddinggemma"

    # Nomic
    "nomic-embed-text"
    "nomic-embed-text-v2-moe"

    # Mixedbread
    "mxbai-embed-large"

    # BAAI
    "bge-m3"
    "bge-large"

    # Snowflake
    "snowflake-arctic-embed:22m"
    "snowflake-arctic-embed:33m"
    "snowflake-arctic-embed:110m"
    "snowflake-arctic-embed:137m"
    "snowflake-arctic-embed:335m"
    "snowflake-arctic-embed2"

    # Sentence Transformers
    "all-minilm:22m"
    "all-minilm:33m"
    "paraphrase-multilingual"

    # IBM Granite
    "granite-embedding:30m"
    "granite-embedding:278m"

    # Jina Embeddings v4 (GGUF Q8_0 via Ollama)
    # v3은 vLLM(fp16), v5는 현재 Ollama/vLLM 미지원 (transformers 5.0 필요)
    "hf.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF:Q8_0"
)

echo "=== Ollama 임베딩 모델 다운로드 ==="
echo "대상: ${#MODELS[@]}개 모델"
echo ""

FAILED=()
for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    if ollama pull "$model"; then
        echo "  ✓ 완료"
    else
        echo "  ✗ 실패"
        FAILED+=("$model")
    fi
    echo ""
done

echo "=== 다운로드 완료 ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "실패한 모델:"
    for m in "${FAILED[@]}"; do
        echo "  - $m"
    done
else
    echo "모든 모델 다운로드 성공"
fi

echo ""
echo "설치된 모델 목록:"
ollama list
