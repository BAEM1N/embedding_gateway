#!/bin/bash

echo "=== Embedding Services 상태 확인 ==="

echo ""
echo "--- Ollama (port 11434) ---"
if curl -s http://localhost:11434/api/version 2>/dev/null; then
    echo ""
    echo "Models:"
    curl -s http://localhost:11434/api/tags | python -m json.tool 2>/dev/null | grep '"name"' || echo "  (no models)"
else
    echo "  OFFLINE"
fi

echo ""
echo "--- TEI (port 8080) ---"
if curl -s http://localhost:8080/health 2>/dev/null; then
    echo ""
    echo "Model info:"
    curl -s http://localhost:8080/info 2>/dev/null | python -m json.tool 2>/dev/null || echo "  (info unavailable)"
else
    echo "  OFFLINE"
fi

echo ""
echo "--- Gateway (port 8000) ---"
if curl -s http://localhost:8000/health 2>/dev/null | python -m json.tool 2>/dev/null; then
    echo ""
    echo "Available models:"
    curl -s http://localhost:8000/v1/models 2>/dev/null | python -m json.tool 2>/dev/null || echo "  (unavailable)"
else
    echo "  OFFLINE"
fi

echo ""
echo "=== 확인 완료 ==="
