# Pre-Flight Checklist ✈️

Before starting hierarchical training, verify everything is ready:

## 📦 Data Files

- [ ] `data/hotpotqa_only/train.parquet` exists
- [ ] `data/hotpotqa_only/test.parquet` exists  
- [ ] `data/hotpotqa_only/corpus.jsonl` exists
- [ ] `data/hotpotqa_only/faiss.index` exists

**Test it:**
```bash
ls -lh data/hotpotqa_only/
```

---

## 🐍 Python Environment

- [ ] Python 3.8+ installed
- [ ] All requirements installed
- [ ] CUDA available

**Test it:**
```bash
python --version  # Should be 3.8+
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 🎮 GPU Resources

- [ ] At least 1 GPU with 24GB+ VRAM
- [ ] GPU is not being used by other processes

**Test it:**
```bash
nvidia-smi
# Check "Memory-Usage" - should have plenty free
```

**Multi-GPU setup (optional but recommended):**
```bash
# If you have 2+ GPUs:
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

---

## 🔧 Configuration Files

- [ ] `launch_retrieval.sh` exists and is executable
- [ ] `launch_searcher.sh` exists and is executable
- [ ] `train_hierarchical_grpo.sh` exists and is executable

**Test it:**
```bash
ls -l launch*.sh train*.sh
# Should show executable permission (x)

# If not executable:
chmod +x launch_retrieval.sh launch_searcher.sh train_hierarchical_grpo.sh
```

---

## 🌐 Network & Ports

- [ ] Port 8000 is available (for Retrieval server)
- [ ] Port 8001 is available (for Searcher server)

**Test it:**
```bash
# Check if ports are free
netstat -tuln | grep 8000
netstat -tuln | grep 8001
# Should return nothing (ports are free)

# If ports are in use, you can change them:
# Edit launch_retrieval.sh: change port 8000 to another port
# Edit launch_searcher.sh: change SEARCHER_PORT to another port
# Edit train_hierarchical_grpo.sh: update retriever.url and hierarchical.searcher_url
```

---

## 📝 WandB (Optional but Recommended)

- [ ] WandB account exists
- [ ] Logged in to WandB

**Test it:**
```bash
wandb login
# Enter your API key when prompted

# Or export it:
export WANDB_API_KEY=your_key_here
```

**Disable WandB (if you prefer console logs only):**
```bash
# Edit train_hierarchical_grpo.sh
# Change: trainer.logger=['wandb']
# To: trainer.logger=['console']
```

---

## 🧪 Quick Service Test

### 1. Test Retrieval Server

**Start it:**
```bash
bash launch_retrieval.sh
```

**In another terminal, test it:**
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is Python?"], "topk": 3, "return_scores": true}'
```

**Expected:** JSON response with 3 documents

**✅ Pass:** Press Ctrl+C to stop server  
**❌ Fail:** Check error messages, verify data files exist

---

### 2. Test Searcher Server

**Start it (after stopping retrieval or in new terminal):**
```bash
# Make sure retrieval server is running first!
bash launch_retrieval.sh  # Terminal 1

# Then in Terminal 2:
bash launch_searcher.sh
```

**In Terminal 3, test it:**
```bash
curl -X POST http://127.0.0.1:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "max_turns": 3}'
```

**Expected:** JSON response like:
```json
{
  "summary": "The capital of France is Paris...",
  "trajectory": ["Initial Query: ...", "Turn 1: ..."],
  "num_searches": 1,
  "success": true
}
```

**✅ Pass:** Both servers working!  
**❌ Fail:** Check error messages in Terminal 2

---

## ✅ All Checks Passed?

If all items above are checked ✅, you're ready to train!

**Start training:**
```bash
# Keep Terminal 1 (Retrieval) and Terminal 2 (Searcher) running
# In Terminal 3:
bash train_hierarchical_grpo.sh
```

**Monitor training:**
- Watch Terminal 3 for training logs
- Check WandB dashboard for metrics
- Checkpoints saved to `verl_checkpoints/`

---

## ❌ Something Failed?

### Common Issues:

**Issue**: Data files not found  
**Fix**: Download or prepare HotpotQA dataset first

**Issue**: CUDA not available  
**Fix**: Install CUDA drivers, verify with `nvidia-smi`

**Issue**: Port in use  
**Fix**: Stop other services or change ports in config files

**Issue**: Out of memory  
**Fix**: Reduce batch size in `train_hierarchical_grpo.sh`

**Issue**: Model download fails  
**Fix**: Check internet connection, authenticate with Hugging Face if needed:
```bash
huggingface-cli login
```

---

## 📚 Need Help?

- **Architecture**: See `RUN_HIERARCHICAL.md`
- **Quick Start**: See `QUICKSTART.md`
- **Configuration**: Check `verl/trainer/config/ppo_trainer.yaml`
- **Issues**: Open a GitHub issue with logs

---

**Ready? Let's train! 🚀**

```bash
# Terminal 1
bash launch_retrieval.sh

# Terminal 2  
bash launch_searcher.sh

# Terminal 3
bash train_hierarchical_grpo.sh
```

