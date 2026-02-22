#!/bin/bash

echo "=== Embedding Services 상태 확인 ==="

echo ""
echo "--- Ollama (port 11434) ---"
if curl -s http://localhost:11434/api/version 2>/dev/null; then
    echo ""
    echo "Models:"
    curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null | grep '"name"' || echo "  (no models)"
else
    echo "  OFFLINE"
fi

echo ""
echo "--- TEI (port 8080) ---"
if curl -s http://localhost:8080/health 2>/dev/null; then
    echo ""
    echo "Model info:"
    curl -s http://localhost:8080/info 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (info unavailable)"
else
    echo "  OFFLINE"
fi

echo ""
echo "--- vLLM (port 8081) ---"
if curl -s http://localhost:8081/health 2>/dev/null; then
    echo ""
    echo "Current model:"
    curl -s http://localhost:8081/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {m[\"id\"]}') for m in d.get('data',[])]" 2>/dev/null || echo "  (info unavailable)"
else
    echo "  OFFLINE"
fi

echo ""
echo "--- Gateway (port 8000) ---"
if curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
    echo ""
    echo "Available models:"
    curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('data', [])
backends = {}
for m in models:
    b = m.get('backend', 'unknown')
    backends.setdefault(b, []).append(m['id'])
for b, ms in sorted(backends.items()):
    print(f'  [{b}] {len(ms)} models')
    for m in sorted(ms)[:5]:
        print(f'    - {m}')
    if len(ms) > 5:
        print(f'    ... and {len(ms)-5} more')
print(f'  Total: {len(models)} models')
" 2>/dev/null || echo "  (unavailable)"
else
    echo "  OFFLINE"
fi

echo ""
echo "=== 확인 완료 ==="
