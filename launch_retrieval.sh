#!/bin/bash
# Launch Retrieval Server
# The retrieval server provides document search functionality

# Configuration
INDEX_PATH="data/hotpotqa_only/faiss.index"
CORPUS_PATH="data/hotpotqa_only/corpus.jsonl"
RETRIEVER_MODEL="intfloat/e5-base-v2"
RETRIEVER_NAME="e5"
TOPK=3
PORT=8000

echo "========================================"
echo "📚 Launching Retrieval Server"
echo "========================================"
echo "Index: $INDEX_PATH"
echo "Corpus: $CORPUS_PATH"
echo "Model: $RETRIEVER_MODEL"
echo "Port: $PORT"
echo "========================================"

python search_r1/search/retrieval_server.py \
    --index_path $INDEX_PATH \
    --corpus_path $CORPUS_PATH \
    --retriever_model $RETRIEVER_MODEL \
    --retriever_name $RETRIEVER_NAME \
    --topk $TOPK


