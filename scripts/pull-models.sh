#!/bin/bash
set -euo pipefail

echo "=== Ollama 임베딩 모델 다운로드 ==="

echo ""
echo "[1/1] bge-m3 (다국어, 567M)..."
ollama pull bge-m3

echo ""
echo "=== 다운로드 완료 ==="
echo ""
echo "추가 모델을 설치하려면:"
echo "  ollama pull snowflake-arctic-embed2"
echo "  ollama pull qwen3-embedding:0.6b"
echo "  ollama pull hf.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF:Q8_0"
echo ""
echo "전체 모델 다운로드:"
echo "  bash scripts/pull-ollama-models.sh"
