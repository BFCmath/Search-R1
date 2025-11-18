# META LOG System - Phase Tracking

## Overview
The META LOG system provides clear announcements for each phase of the training process, making it easy to track what's happening at any moment.

## Phase Markers

All META LOG messages follow this format:
```
================================================================================
[EMOJI] [META] PHASE: [PHASE_NAME]
================================================================================
```

---

## Complete Phase Flow

### 1. 🔧 INITIALIZATION
**When**: During system startup, before any training or validation
**What**: Creating resource pools, worker groups, and initializing models

```
================================================================================
🔧 [META] PHASE: INITIALIZATION - Creating Resource Pools and Workers
================================================================================
```

**Activities**:
- Creating Ray resource pools
- Initializing actor, rollout, and reference policy workers
- Loading model weights
- Setting up FSDP/distributed training

---

### 2. 🧪 PRE-TRAINING VALIDATION
**When**: Before training starts (if `val_before_train: true`)
**What**: Running validation on the entire validation set with initial model

```
================================================================================
🧪 [META] PHASE: PRE-TRAINING VALIDATION
================================================================================

[... validation runs ...]

================================================================================
🔍 [META] PHASE: VALIDATION
================================================================================
Total batches: 28 | Total questions: 7405 | Batch size: 256
================================================================================

📊 Validation [1/28] (3.6%) | Questions: 256/7405 | Avg Reward: 0.456 | ...
📊 Validation [2/28] (7.1%) | Questions: 512/7405 | Avg Reward: 0.489 | ...
...

================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================
Total questions processed: 7405 | Total time: 19.2m | Avg time per batch: 41.1s
================================================================================
```

**Special Case**: If `val_only: true`, training exits here:
```
================================================================================
🏁 [META] PHASE: COMPLETE - Validation Only Mode
================================================================================
```

---

### 3. 🚀 TRAINING START
**When**: After pre-training validation completes
**What**: Beginning of the main training loop

```
================================================================================
🚀 [META] PHASE: TRAINING START
================================================================================
Total epochs: 3 | Total steps: 35
================================================================================
```

---

### 4. 🔄 TRAINING STEP
**When**: Each training step (repeated for every batch)
**What**: Processing one training batch through the PPO pipeline

```
================================================================================
🔄 [META] PHASE: TRAINING STEP
[Epoch 1/3] Step 1/35 (2.9%) | Elapsed: 42s | ETA: 23.8m
================================================================================

ACTIVE_TRAJ_NUM: [512, 287, 45, 23]
📊 Metrics: finish_ratio=0.955 | gen=11.234 | kl=0.012 | loss=0.234 | ...
[Step 1] mean=0.782 kl=0.012 loss=0.234 finish_ratio=0.955 ...
```

**Activities per step**:
- Generation phase (LLM + search loop)
- Reference policy log prob computation
- Value computation (if using critic)
- Advantage estimation
- Actor/Critic updates
- Metric collection

---

### 5. 🧪 PERIODIC VALIDATION
**When**: Every `test_freq` steps (e.g., every 50 steps)
**What**: Running validation during training to monitor progress

```
================================================================================
🧪 [META] PHASE: PERIODIC VALIDATION (Step 50)
================================================================================

[... validation runs with detailed progress ...]

================================================================================
🔍 [META] PHASE: VALIDATION
================================================================================
Total batches: 28 | Total questions: 7405 | Batch size: 256
================================================================================

📊 Validation [1/28] (3.6%) | Questions: 256/7405 | Avg Reward: 0.623 | ...
...

================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================
```

---

### 6. 💾 CHECKPOINT SAVING
**When**: Every `save_freq` steps (e.g., every 100 steps)
**What**: Saving model checkpoints to disk

```
================================================================================
💾 [META] PHASE: CHECKPOINT SAVING (Step 100)
================================================================================

[... saving checkpoint files ...]

================================================================================
✅ [META] CHECKPOINT SAVED SUCCESSFULLY
================================================================================
```

**What's saved**:
- Actor model weights
- Critic model weights (if using GAE)
- Optimizer states
- Training metadata

---

### 7. 🎉 TRAINING COMPLETE
**When**: After all training steps are finished
**What**: End of main training loop

```
================================================================================
🎉 [META] PHASE: TRAINING COMPLETE
================================================================================
Total steps completed: 35 | Total time: 25.6m
================================================================================
```

---

### 8. 🧪 POST-TRAINING VALIDATION
**When**: After training completes (if validation is enabled)
**What**: Final validation with trained model

```
================================================================================
🧪 [META] PHASE: POST-TRAINING VALIDATION
================================================================================

[... validation runs ...]

================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================
```

---

### 9. 🏁 ALL PHASES COMPLETE
**When**: End of entire training pipeline
**What**: Final completion message

```
================================================================================
🏁 [META] ALL PHASES COMPLETE - Training Finished Successfully!
================================================================================
```

---

## Quick Reference Table

| Emoji | Phase | Frequency | Description |
|-------|-------|-----------|-------------|
| 🔧 | INITIALIZATION | Once (start) | Setting up workers and models |
| 🧪 | PRE-TRAINING VALIDATION | Once (before training) | Initial model validation |
| 🚀 | TRAINING START | Once | Beginning of training |
| 🔄 | TRAINING STEP | Every step | Individual training iteration |
| 🧪 | PERIODIC VALIDATION | Every `test_freq` steps | Mid-training validation |
| 💾 | CHECKPOINT SAVING | Every `save_freq` steps | Model checkpoint save |
| 🎉 | TRAINING COMPLETE | Once (end of training) | Training finished |
| 🧪 | POST-TRAINING VALIDATION | Once (after training) | Final validation |
| 🏁 | ALL PHASES COMPLETE | Once (final) | Everything finished |

---

## Configuration Impact

Your `train_grpo.sh` config affects which phases run:

```bash
trainer.val_before_train=true    # Enables PRE-TRAINING VALIDATION
trainer.val_only=false           # If true, exits after PRE-TRAINING VALIDATION
trainer.test_freq=50             # PERIODIC VALIDATION every 50 steps
trainer.save_freq=100            # CHECKPOINT SAVING every 100 steps
trainer.total_training_steps=35  # Total TRAINING STEP iterations
```

---

## Example: Complete Training Run

```bash
# STARTUP
================================================================================
🔧 [META] PHASE: INITIALIZATION - Creating Resource Pools and Workers
================================================================================
[... model loading ...]

# PRE-TRAINING VALIDATION
================================================================================
🧪 [META] PHASE: PRE-TRAINING VALIDATION
================================================================================
[... validation with 28 batches ...]
================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================

# TRAINING BEGINS
================================================================================
🚀 [META] PHASE: TRAINING START
================================================================================
Total epochs: 3 | Total steps: 35
================================================================================

# STEP 1
================================================================================
🔄 [META] PHASE: TRAINING STEP
[Epoch 1/3] Step 1/35 (2.9%) | Elapsed: 42s | ETA: 23.8m
================================================================================
[... training step 1 ...]

# STEP 2
================================================================================
🔄 [META] PHASE: TRAINING STEP
[Epoch 1/3] Step 2/35 (5.7%) | Elapsed: 1.5m | ETA: 24.2m
================================================================================
[... training step 2 ...]

# ... (steps 3-49) ...

# STEP 50 - PERIODIC VALIDATION
================================================================================
🔄 [META] PHASE: TRAINING STEP
[Epoch 1/3] Step 50/35 (...)
================================================================================
[... training step 50 ...]

================================================================================
🧪 [META] PHASE: PERIODIC VALIDATION (Step 50)
================================================================================
[... validation ...]
================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================

# ... (more training steps) ...

# STEP 100 - CHECKPOINT SAVE
================================================================================
💾 [META] PHASE: CHECKPOINT SAVING (Step 100)
================================================================================
[... saving ...]
================================================================================
✅ [META] CHECKPOINT SAVED SUCCESSFULLY
================================================================================

# ... (continue to step 35) ...

# TRAINING COMPLETES
================================================================================
🎉 [META] PHASE: TRAINING COMPLETE
================================================================================
Total steps completed: 35 | Total time: 25.6m
================================================================================

# POST-TRAINING VALIDATION
================================================================================
🧪 [META] PHASE: POST-TRAINING VALIDATION
================================================================================
[... final validation ...]
================================================================================
✅ [META] PHASE COMPLETE: VALIDATION
================================================================================

# ALL DONE
================================================================================
🏁 [META] ALL PHASES COMPLETE - Training Finished Successfully!
================================================================================
```

---

## Benefits

✅ **Clear Context**: Always know what phase you're in
✅ **Easy Debugging**: Quickly identify where issues occur
✅ **Progress Tracking**: See major milestones at a glance
✅ **Log Parsing**: Easy to grep/search for specific phases
✅ **Visual Clarity**: Emoji + formatting makes scanning logs easy

---

## Searching Logs

Use these grep patterns to find specific phases:

```bash
# Find all phase transitions
grep "\[META\]" training.log

# Find all validation phases
grep "\[META\] PHASE.*VALIDATION" training.log

# Find all training steps
grep "\[META\] PHASE: TRAINING STEP" training.log

# Find checkpoint saves
grep "CHECKPOINT" training.log

# Find completion
grep "COMPLETE" training.log
```

---

## Integration with Existing Logs

META LOG announcements work **alongside** existing output:
- Ray worker logs (with `[36m` color codes)
- ACTIVE_TRAJ_NUM statistics
- Model generation samples
- Warning messages
- Metric summaries

The META LOG system **adds structure** without removing any existing information!

