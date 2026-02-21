#!/bin/bash
# WSL2 Ubuntu 내부에서 실행하세요.
# Docker Engine + NVIDIA Container Toolkit 설치 스크립트
#
# 사전 조건:
#   1. Windows에 NVIDIA 드라이버가 설치되어 있어야 합니다.
#   2. WSL2 Ubuntu가 설치되어 있어야 합니다: wsl --install -d Ubuntu-24.04
#   3. WSL2 내부에서 이 스크립트를 실행하세요.
#
# 주의: WSL2 내부에 NVIDIA Linux 드라이버를 설치하지 마세요!
#       Windows 호스트 드라이버가 자동으로 WSL2에 노출됩니다.

set -euo pipefail

echo "=== WSL2 Docker + NVIDIA Container Toolkit 설치 ==="
echo ""

# 1. Docker Engine 설치
echo "[1/4] Docker Engine 설치 중..."
if command -v docker &> /dev/null; then
    echo "Docker가 이미 설치되어 있습니다: $(docker --version)"
else
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh
    sudo usermod -aG docker "$USER"
    echo "Docker 설치 완료. 그룹 변경 적용을 위해 재로그인이 필요할 수 있습니다."
fi

# 2. Docker 서비스 시작
echo ""
echo "[2/4] Docker 서비스 시작 중..."
sudo service docker start || sudo systemctl start docker

# 3. NVIDIA Container Toolkit 설치
echo ""
echo "[3/4] NVIDIA Container Toolkit 설치 중..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "NVIDIA Container Toolkit이 이미 설치되어 있습니다."
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo service docker restart || sudo systemctl restart docker
    echo "NVIDIA Container Toolkit 설치 완료."
fi

# 4. GPU 접근 검증
echo ""
echo "[4/4] GPU 접근 검증 중..."
echo ""
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "이제 다음 명령어로 TEI를 실행할 수 있습니다:"
echo "  docker compose -f /mnt/c/Users/baeumai/embedding/docker/docker-compose.yml up -d"
echo ""
echo "Windows Git Bash에서 WSL2 Docker를 사용하려면:"
echo "  export DOCKER_HOST=unix:///mnt/wsl/docker.sock"
echo "  또는"
echo "  wsl docker compose -f docker/docker-compose.yml up -d"
