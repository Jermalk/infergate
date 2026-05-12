#!/usr/bin/env bash
# Three demo requests that exercise signal, embedding, and keyword routing.
# Gateway must be running: ./run_demo.sh
set -euo pipefail

GW="${INFERGATE_GW:-http://localhost:8090}"

hr() { printf '\n%s\n' "────────────────────────────────────────"; }

call() {
    local label="$1"
    local payload="$2"
    hr
    echo ">>> $label"
    local response
    response=$(curl -s -X POST "$GW/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --dump-header /tmp/infergate_demo_headers.txt)
    grep "X-InferGate-" /tmp/infergate_demo_headers.txt || true
    echo "$response" | python3 -c "
import sys, json
try:
    obj = json.load(sys.stdin)
    choices = obj.get('choices', [])
    if choices:
        msg = choices[0].get('message', {})
        text = msg.get('content') or msg.get('reasoning_content', '')
        print('reply:', (text or '').strip()[:200])
    else:
        print('response:', json.dumps(obj)[:200])
except Exception as e:
    print('parse error:', e, sys.stdin.read()[:200])
" 2>/dev/null
}

# 1. Signal routing — #code directive fires O(1) signal, bypasses embedding
#    → task_class=code, strategy=signal, backend=ollama (local/fast tier)
call "1 — signal: #code directive (O1, no embedding)" \
'{"messages":[{"role":"user","content":"#code Write a Python one-liner to flatten a nested list."}],"max_tokens":80}'

# 2. Embedding routing — plain question with no keywords, classified by cosine similarity
#    → task_class=general, strategy=embedding, backend=ollama
call "2 — embedding: general question (cosine similarity classification)" \
'{"messages":[{"role":"user","content":"What is the speed of light in metres per second?"}],"max_tokens":64}'

# 3. Keyword routing — "implement" keyword triggers code task class via keyword signal
#    → task_class=code, strategy=signal, backend=ollama
call "3 — keyword: 'implement' keyword match -> code task class" \
'{"messages":[{"role":"user","content":"Implement a binary search in Python."}],"max_tokens":96}'
