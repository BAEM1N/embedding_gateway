#!/bin/bash
# TEI 모델 weight를 미리 다운로드하는 스크립트.
# 각 모델에 대해 TEI 컨테이너를 잠깐 실행하여 weight를 볼륨에 캐시한 뒤 종료.
# 이후 모델 교체 시 네트워크 없이 캐시에서 바로 로딩 가능.
#
# 사용법:
#   bash scripts/prefetch-tei-models.sh              # 전체 모델 다운로드
#   bash scripts/prefetch-tei-models.sh "model-id"   # 특정 모델만 다운로드

set -euo pipefail

IMAGE="ghcr.io/huggingface/text-embeddings-inference:89-1.9"
CONTAINER="tei-prefetch"
WSL_DISTRO="Ubuntu-24.04"

# HF_TOKEN 환경변수에서 읽기 (.env 파일 또는 직접 설정)
HF_TOKEN="${HF_TOKEN:-}"

MODELS=(
    "BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "Qwen/Qwen3-Embedding-4B"
    "google/embeddinggemma-300m"
    "nomic-ai/nomic-embed-text-v1.5"
    "nomic-ai/nomic-embed-text-v2-moe"
    "Snowflake/snowflake-arctic-embed-l-v2.0"
    "intfloat/multilingual-e5-base"
    "intfloat/multilingual-e5-large-instruct"
    "nlpai-lab/KURE-v1"
)

# 특정 모델만 다운로드할 경우
if [ $# -ge 1 ]; then
    MODELS=("$@")
fi

docker_cmd() {
    wsl -d "$WSL_DISTRO" -- docker "$@"
}

echo "=== TEI 모델 사전 다운로드 ==="
echo "대상 모델: ${#MODELS[@]}개"

if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: 설정됨 (gated 모델 접근 가능)"
else
    echo "HF_TOKEN: 미설정 (gated 모델은 다운로드 실패할 수 있음)"
fi

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- $model ---"

    # 기존 컨테이너 정리
    docker_cmd rm -f "$CONTAINER" 2>/dev/null || true

    # HF 토큰 환경변수 설정
    ENV_ARGS=()
    if [ -n "$HF_TOKEN" ]; then
        ENV_ARGS=(-e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN")
    fi

    # 모델 다운로드용 컨테이너 시작
    echo "  다운로드 시작..."
    docker_cmd run -d \
        --name "$CONTAINER" \
        --gpus all \
        -v tei-model-cache:/data \
        "${ENV_ARGS[@]}" \
        "$IMAGE" \
        --model-id "$model" \
        --dtype float16

    # /health가 200을 반환할 때까지 대기 (= 모델 로딩 완료)
    echo "  모델 로딩 대기..."
    LOADED=false
    for i in $(seq 1 120); do
        if docker_cmd exec "$CONTAINER" curl -sf http://localhost:80/health > /dev/null 2>&1; then
            echo "  ✓ 완료."
            LOADED=true
            break
        fi

        # 컨테이너가 종료되었는지 확인 (다운로드 실패 등)
        STATUS=$(docker_cmd inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "exited" ]; then
            echo "  ✗ 컨테이너 종료됨. 로그:"
            docker_cmd logs --tail 20 "$CONTAINER"
            break
        fi

        sleep 3
    done

    if [ "$LOADED" = false ] && [ "$STATUS" != "exited" ]; then
        echo "  ✗ 타임아웃 (360초). 로그:"
        docker_cmd logs --tail 10 "$CONTAINER"
    fi

    # 컨테이너 종료 (볼륨에 캐시 유지)
    docker_cmd rm -f "$CONTAINER" 2>/dev/null || true
done

echo ""
echo "=== 사전 다운로드 완료 ==="
echo "캐시된 모델은 tei-model-cache 볼륨에 저장되어 있습니다."
