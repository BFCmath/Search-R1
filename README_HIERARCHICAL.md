# Hierarchical Thinker-Searcher Framework

A two-tier reinforcement learning framework for question answering with search capabilities.

## 🎯 What is This?

Instead of training a single agent to both reason and search, we split responsibilities:

- **Thinker (0.5B)**: Small model that does high-level reasoning and delegates searches
- **Searcher (1.5B)**: Larger model that performs detailed multi-turn search loops

**Key Innovation**: Only the Thinker is trained with RL. The Searcher stays frozen, so we get better search capabilities without the computational cost of training a large model.

## 🏗️ Architecture

```
User Question
     ↓
┌────────────────────┐
│   Thinker (0.5B)   │  ← Trained with RL
│   - Reasoning      │
│   - Delegation     │
└─────────┬──────────┘
          │ <search>query</search>
          ↓
┌────────────────────┐
│  Searcher (1.5B)   │  ← Frozen, runs as service
│  - Multi-turn      │
│  - Search loops    │
│  - Summarization   │
└─────────┬──────────┘
          │ <information>summary</information>
          ↓
┌────────────────────┐
│   Thinker (0.5B)   │
│   - Final answer   │
└────────────────────┘
```

## 🚀 Quick Start

**1. Check prerequisites:**
```bash
cat PRE_FLIGHT_CHECK.md  # Review and verify all requirements
```

**2. Launch in 3 terminals:**

**Terminal 1** - Retrieval:
```bash
bash launch_retrieval.sh
```

**Terminal 2** - Searcher:
```bash
bash launch_searcher.sh
```

**Terminal 3** - Training:
```bash
bash train_hierarchical_grpo.sh
```

**3. Monitor progress:**
- Console logs in Terminal 3
- WandB dashboard: Project "Search-R1-Hierarchical"
- Checkpoints: `verl_checkpoints/hotpotqa-hierarchical-thinker0.5b-searcher1.5b/`

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - TL;DR quick start guide
- **[PRE_FLIGHT_CHECK.md](PRE_FLIGHT_CHECK.md)** - Verify everything before running
- **[RUN_HIERARCHICAL.md](RUN_HIERARCHICAL.md)** - Detailed usage and configuration

## 🔧 Configuration

### Models
- **Thinker**: `Qwen/Qwen2.5-0.5B-Instruct` (trained)
- **Searcher**: `Qwen/Qwen2.5-1.5B-Instruct` (frozen)

### Key Parameters
```yaml
hierarchical.thinker_max_turns: 2          # Thinker reasoning iterations
hierarchical.searcher_max_turns: 3         # Searcher search loops
data.train_batch_size: 256                 # Training batch size
trainer.total_epochs: 3                    # Total epochs
actor_rollout_ref.actor.state_masking: true  # Mask searcher tokens
```

Edit `train_hierarchical_grpo.sh` or `verl/trainer/config/ppo_trainer.yaml` to customize.

## 📊 Results

The framework trains the Thinker to:
1. Recognize when it needs more information
2. Formulate precise search queries
3. Delegate to the Searcher service
4. Integrate Searcher results into reasoning
5. Produce accurate final answers

**Metrics tracked:**
- Exact Match (EM) accuracy
- Search delegation frequency
- Valid action ratio
- Reward/score trends

## 🐛 Troubleshooting

**"Cannot connect to Searcher service"**
→ Make sure `launch_searcher.sh` is running in Terminal 2

**CUDA Out of Memory**
→ Reduce batch size or use separate GPUs for Searcher and Thinker

**Port already in use**
→ Change ports in `launch_retrieval.sh` and `launch_searcher.sh`

See [RUN_HIERARCHICAL.md](RUN_HIERARCHICAL.md) for detailed troubleshooting.

## 🔬 How It Works

### Token Masking
The Searcher's output (summaries in `<information>` tags) is masked during training:
- Thinker tokens: **Trained with RL gradients** ✅
- Searcher tokens: **Masked (no gradients)** ⛔

This ensures only the Thinker learns delegation and reasoning, while benefiting from a stronger frozen Searcher.

### Training Loop
1. Thinker generates `<thinking>` and `<search>query</search>`
2. Query sent to Searcher service (HTTP POST)
3. Searcher performs 1-3 search-observe loops
4. Searcher returns `<answer>summary</answer>`
5. Summary wrapped in `<information>` tags for Thinker
6. Thinker continues reasoning or produces `<answer>`
7. RL update applied to **Thinker tokens only**

## 📈 Advantages

### vs. Standard Search-R1:
- ✅ Better search (1.5B Searcher vs single API call)
- ✅ Cheaper training (only train 0.5B, not 1.5B)
- ✅ Modular (can upgrade Searcher independently)
- ✅ Specialized roles (delegation + search)

### vs. Single Large Model:
- ✅ More efficient (0.5B + frozen 1.5B vs training 3B+)
- ✅ Easier to train (smaller model, cleaner gradients)
- ✅ Scalable (can run multiple Searcher instances)

## 📁 Project Structure

```
.
├── launch_retrieval.sh              # Start retrieval server
├── launch_searcher.sh               # Start Searcher agent
├── train_hierarchical_grpo.sh       # Train Thinker with GRPO
├── search_r1/
│   ├── llm_agent/
│   │   └── hierarchical_generation.py   # Thinker-Searcher orchestration
│   └── search/
│       ├── retrieval_server.py          # Document retrieval API
│       └── searcher_server.py           # Searcher agent service
├── verl/
│   └── trainer/
│       ├── main_ppo_hierarchical.py     # Training entry point
│       ├── config/ppo_trainer.yaml      # Configuration
│       └── ppo/ray_trainer.py           # Training logic
└── data/
    └── hotpotqa_only/
        ├── train.parquet                # Training questions
        ├── test.parquet                 # Test questions
        ├── corpus.jsonl                 # Documents
        └── faiss.index                  # Search index
```

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{search-r1-hierarchical,
  title={Hierarchical Thinker-Searcher Framework for Multi-Hop Question Answering},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Search-R1}
}
```

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📮 Contact

Questions? Issues? Open a GitHub issue or contact [your email].

---

**Ready to train?** Start with [QUICKSTART.md](QUICKSTART.md)!

