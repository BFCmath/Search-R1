#!/bin/bash
# Launch Searcher Agent Server
# The Searcher (Qwen2.5-1.5B) runs as a separate service

SEARCHER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SEARCHER_PORT=8001
RETRIEVAL_URL="http://127.0.0.1:8000/retrieve"
MAX_TURNS=3
TOPK=3

echo "========================================"
echo "🤖 Launching Searcher Agent Server"
echo "========================================"
echo "Model: $SEARCHER_MODEL"
echo "Port: $SEARCHER_PORT"
echo "Max Turns: $MAX_TURNS"
echo "Retrieval URL: $RETRIEVAL_URL"
echo "========================================"

python search_r1/search/searcher_server.py \
    --model_path $SEARCHER_MODEL \
    --port $SEARCHER_PORT \
    --max_turns $MAX_TURNS \
    --retrieval_url $RETRIEVAL_URL \
    --topk $TOPK

