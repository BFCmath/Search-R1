# How to Run Hierarchical Thinker-Searcher Training

## Quick Start

### Step 1: Prepare Data
Make sure your HotpotQA data is in the correct location:
```bash
ls data/hotpotqa_only/
# Should show: train.parquet test.parquet
```

### Step 2: Launch Retrieval Server (Terminal 1)
```bash
# Start the document retrieval server on port 8000
bash launch_retrieval.sh
```

This will load the FAISS index and corpus, then start the retrieval API.

Wait until you see: `INFO:     Application startup complete.`

### Step 3: Launch Searcher Server (Terminal 2)
```bash
# Start the Searcher agent service on port 8001
bash launch_searcher.sh
```

This will:
- Load Qwen2.5-1.5B-Instruct model
- Start FastAPI server on port 8001
- Connect to retrieval server at localhost:8000

Wait until you see: `🚀 [SEARCHER SERVER] Starting on http://0.0.0.0:8001`

### Step 4: Test Searcher Service (Optional)
```bash
# In a new terminal, test that Searcher is working
curl -X POST http://127.0.0.1:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "max_turns": 3}'
```

You should get a JSON response with a summary answer.

### Step 5: Start Training (Terminal 3)
```bash
# Train the Thinker model with hierarchical GRPO
bash train_hierarchical_grpo.sh
```

This will:
- Load Thinker model (Qwen2.5-0.5B-Instruct)
- Connect to Searcher service
- Start RL training with GRPO
- Log to WandB project "Search-R1-Hierarchical"
- Save checkpoints to `verl_checkpoints/hotpotqa-hierarchical-thinker0.5b-searcher1.5b/`

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Terminal 1: Retrieval Server (Port 8000)                    │
│ - Serves Wikipedia documents                                 │
│ - Used by Searcher to get search results                     │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ HTTP requests for documents
                              │
┌─────────────────────────────────────────────────────────────┐
│ Terminal 2: Searcher Server (Port 8001)                     │
│ - Qwen2.5-1.5B-Instruct (Frozen)                            │
│ - Performs multi-turn search (max 3 turns)                  │
│ - Returns summaries to Thinker                               │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ HTTP requests with search queries
                              │
┌─────────────────────────────────────────────────────────────┐
│ Terminal 3: Training Process                                 │
│ - Qwen2.5-0.5B-Instruct (RL Trained)                        │
│ - Delegates searches to Searcher service                     │
│ - Learns via GRPO algorithm                                  │
└─────────────────────────────────────────────────────────────┘
```

## Training Parameters

### Models
- **Thinker**: `Qwen/Qwen2.5-0.5B-Instruct` (trained)
- **Searcher**: `Qwen/Qwen2.5-1.5B-Instruct` (frozen)

### Configuration (from `train_hierarchical_grpo.sh`)
```bash
# Thinker config
hierarchical.thinker_max_turns=2                    # Max 2 reasoning iterations
hierarchical.thinker_max_response_length=300        # 300 tokens per response

# Searcher config  
hierarchical.searcher_max_turns=3                   # Max 3 search-observe loops
hierarchical.searcher_max_response_length=400       # 400 tokens per response
hierarchical.searcher_max_obs_length=400            # 400 tokens for observations

# Training config
data.train_batch_size=256                           # Batch size for training
trainer.total_epochs=3                              # Number of epochs
trainer.total_training_steps=35                     # Total steps
actor_rollout_ref.actor.state_masking=true         # Mask Searcher tokens
```

## Monitoring Training

### WandB
Training metrics are logged to WandB project `Search-R1-Hierarchical`:
- Rewards and scores
- Valid action ratio
- Search frequency
- Loss values

### Console Output
You'll see detailed logs like:
```
🧠 [Thinker Turn 1/2] Active: 256/256
  Sample 0: 🔍 Thinker search → Delegating to Searcher
    Query: What is the population of Paris?
    ✅ Searcher returned: Paris has a population of approximately 2.1 million...
  Sample 1: ✅ Thinker answered
```

### Checkpoints
Checkpoints are saved every 100 steps to:
```
verl_checkpoints/hotpotqa-hierarchical-thinker0.5b-searcher1.5b/actor/global_step_X/
```

## Validation

The training script will run validation:
- **Before training** (`val_before_train=true`)
- **Every 50 steps** (`test_freq=50`)
- **After training** (automatically)

Validation computes exact match (EM) scores on the test set.

## Common Issues

### Issue: "Cannot connect to Searcher service"
**Solution**: Make sure Searcher server is running on port 8001
```bash
# Check if server is running
curl http://127.0.0.1:8001/health

# If not, restart it
bash launch_searcher.sh
```

### Issue: "Cannot connect to retrieval server"
**Solution**: Make sure retrieval server is running on port 8000
```bash
# Test if server is running
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test"], "topk": 1}'

# Start it if needed
bash launch_retrieval.sh
```

### Issue: CUDA out of memory
**Solution**: Adjust GPU memory allocation in `train_hierarchical_grpo.sh`:
```bash
# Reduce Thinker rollout GPU usage
actor_rollout_ref.rollout.gpu_memory_utilization=0.35  # Try 0.25

# Reduce Searcher GPU usage in launch_searcher.sh
# Or run Searcher on a different GPU by setting CUDA_VISIBLE_DEVICES
```

### Issue: Port already in use
**Solution**: Change ports in the scripts:
```bash
# In launch_searcher.sh
SEARCHER_PORT=8002  # Instead of 8001

# In train_hierarchical_grpo.sh
hierarchical.searcher_url="http://127.0.0.1:8002/search"
```

## Stopping Training

To gracefully stop training:
1. Press `Ctrl+C` in Terminal 3 (training process)
2. Wait for checkpoint to save
3. `Ctrl+C` in Terminal 2 (Searcher server)
4. `Ctrl+C` in Terminal 1 (Retrieval server)

Checkpoints are auto-saved, so you can resume training from the last checkpoint.

## Using Trained Model

After training, inference with the trained Thinker:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# Load trained Thinker
model_path = "verl_checkpoints/hotpotqa-hierarchical-thinker0.5b-searcher1.5b/actor/global_step_35"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create prompt
question = "What is the capital of France and what is its population?"
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

# Generate with Thinker (it will call Searcher service automatically via HTTP)
# Note: Searcher service must still be running!
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0]))
```

## Advanced: Running on Multiple GPUs

### Option 1: Separate GPUs for Thinker and Searcher
```bash
# Terminal 2: Run Searcher on GPU 1
export CUDA_VISIBLE_DEVICES=1
bash launch_searcher.sh

# Terminal 3: Run Thinker training on GPU 0
export CUDA_VISIBLE_DEVICES=0
bash train_hierarchical_grpo.sh
```

### Option 2: Multi-node Training
Edit `train_hierarchical_grpo.sh`:
```bash
trainer.nnodes=2                    # Number of nodes
trainer.n_gpus_per_node=4           # GPUs per node
```

## Key Differences from Standard Search-R1

| Feature | Standard Search-R1 | Hierarchical |
|---------|-------------------|--------------|
| Model | Single agent (1.5B) | Thinker (0.5B) + Searcher (1.5B) |
| Search | Direct retrieval | Multi-turn via Searcher agent |
| Training | All tokens trained | Only Thinker trained (Searcher masked) |
| Deployment | Single process | Two services (Thinker + Searcher) |
| Complexity | Lower | Higher (but more flexible) |

## Next Steps

After training completes:
1. Check validation scores in WandB
2. Analyze which samples benefit from Searcher delegation
3. Experiment with different Thinker/Searcher model sizes
4. Try other datasets (2WikiMultihopQA, MuSiQue, etc.)
5. Tune hyperparameters (learning rate, KL coefficient, etc.)

Happy training! 🚀

