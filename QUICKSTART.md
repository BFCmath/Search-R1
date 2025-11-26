# Hierarchical Thinker-Searcher - Quick Start

## TL;DR - Run in 3 Terminals

**Terminal 1** - Start retrieval server:
```bash
bash launch_retrieval.sh
```

**Terminal 2** - Start Searcher agent:
```bash
bash launch_searcher.sh
```

**Terminal 3** - Start training:
```bash
bash train_hierarchical_grpo.sh
```

That's it! Training will start and you'll see logs from all three components.

---

## What's Happening?

### 🔹 Terminal 1: Retrieval Server (Port 8000)
- Loads FAISS index for fast document search
- Serves Wikipedia/corpus documents
- Used by the Searcher to find relevant information

### 🔹 Terminal 2: Searcher Agent (Port 8001)  
- Loads **Qwen2.5-1.5B-Instruct** model (frozen, not trained)
- Receives search queries from Thinker
- Performs multi-turn search loops (up to 3 turns)
- Returns summarized answers

### 🔹 Terminal 3: RL Training
- Trains **Qwen2.5-0.5B-Instruct** model (Thinker)
- Thinker learns to delegate searches to Searcher
- Only Thinker tokens are trained (Searcher outputs masked)
- Logs to WandB: `Search-R1-Hierarchical`

---

## Prerequisites

### 1. Data Files
Ensure you have the HotpotQA dataset prepared:
```
data/hotpotqa_only/
├── train.parquet          # Training questions
├── test.parquet           # Test questions  
├── corpus.jsonl           # Document corpus
└── faiss.index            # FAISS index for retrieval
```

If missing, prepare the data first (see data preparation guide).

### 2. GPU Requirements
- **Minimum**: 1 GPU with 24GB VRAM (e.g., RTX 3090, A5000)
- **Recommended**: 2 GPUs
  - GPU 0: Training (Thinker)
  - GPU 1: Searcher service

### 3. Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Expected Output

### When Retrieval Server Starts:
```
========================================
📚 Launching Retrieval Server
========================================
Loading FAISS index...
Loading corpus...
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### When Searcher Server Starts:
```
========================================
🤖 [SEARCHER] Loading model: Qwen/Qwen2.5-1.5B-Instruct
========================================
✅ [SEARCHER] Model loaded successfully!
🚀 [SEARCHER SERVER] Starting on http://0.0.0.0:8001
INFO:     Application startup complete.
```

### When Training Starts:
```
================================================================================
🧠 [HIERARCHICAL MODE] Enabled
================================================================================
Thinker model: Qwen/Qwen2.5-0.5B-Instruct
Thinker max turns: 2
Searcher model: Qwen/Qwen2.5-1.5B-Instruct
Searcher max turns: 3
================================================================================

🔄 [META] PHASE: TRAINING STEP
[Epoch 1/3] Step 1/35 (2.9%)
================================================================================

🧠 [Thinker Turn 1/2] Active: 256/256
  Sample 0: 🔍 Thinker search → Delegating to Searcher
    Query: What is the population of Paris?
    ✅ Searcher returned: Paris has a population of approximately 2.1 million...
```

---

## Monitoring Progress

### 1. Console Logs
Real-time training metrics in Terminal 3:
- Rewards (mean, max, min)
- Valid action ratio
- Search frequency
- Timing per step

### 2. WandB Dashboard
Open your WandB project to see:
- Training curves (rewards, loss, KL divergence)
- Validation accuracy (exact match)
- System metrics (GPU usage, timing)

### 3. Checkpoints
Models saved every 100 steps to:
```
verl_checkpoints/hotpotqa-hierarchical-thinker0.5b-searcher1.5b/
└── actor/
    ├── global_step_100/
    ├── global_step_200/
    └── ...
```

---

## Stopping Training

**Graceful shutdown:**
1. Press `Ctrl+C` in Terminal 3 (training)
2. Wait for checkpoint to save
3. Press `Ctrl+C` in Terminal 2 (Searcher)
4. Press `Ctrl+C` in Terminal 1 (Retrieval)

**Resume training** from checkpoint:
```bash
# Edit train_hierarchical_grpo.sh
# Add checkpoint path to config

# Then run normally
bash train_hierarchical_grpo.sh
```

---

## Troubleshooting

### ❌ Error: "Cannot connect to Searcher service at http://127.0.0.1:8001/search"

**Cause**: Searcher server not running or crashed

**Fix**:
```bash
# Check Searcher server logs in Terminal 2
# Look for errors or OOM messages

# Restart Searcher
bash launch_searcher.sh
```

### ❌ Error: "Search failed due to connection error"

**Cause**: Retrieval server not responding

**Fix**:
```bash
# Test retrieval server
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test"], "topk": 1}'

# Should return JSON with documents
# If not, check Terminal 1 for errors
```

### ❌ CUDA Out of Memory

**Option 1**: Reduce batch size
```bash
# In train_hierarchical_grpo.sh
data.train_batch_size=128  # Instead of 256
```

**Option 2**: Reduce GPU memory
```bash
# In train_hierarchical_grpo.sh
actor_rollout_ref.rollout.gpu_memory_utilization=0.25  # Instead of 0.35

# In launch_searcher.sh (if needed, add this line)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Option 3**: Use separate GPUs
```bash
# Terminal 2: Searcher on GPU 1
export CUDA_VISIBLE_DEVICES=1
bash launch_searcher.sh

# Terminal 3: Training on GPU 0  
export CUDA_VISIBLE_DEVICES=0
bash train_hierarchical_grpo.sh
```

---

## Next Steps

After training completes:

1. **Evaluate** the trained model on test set
2. **Compare** with baseline (standard Search-R1)
3. **Analyze** when Thinker delegates to Searcher
4. **Tune** hyperparameters (learning rate, KL coef, etc.)
5. **Scale** to larger models or datasets

---

## Full Documentation

For detailed information:
- Architecture details → `RUN_HIERARCHICAL.md`
- Configuration options → `verl/trainer/config/ppo_trainer.yaml`
- Code overview → `search_r1/llm_agent/hierarchical_generation.py`

---

**Questions?** Check the troubleshooting section or open an issue!

Happy training! 🚀

