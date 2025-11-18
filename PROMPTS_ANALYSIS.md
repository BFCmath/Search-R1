# Prompt Analysis: Thinker vs Searcher

## 📋 Overview

This document shows the exact prompts used for both the **Thinker** (0.5B) and **Searcher** (1.5B) models in the hierarchical framework.

---

## 🧠 THINKER Model Prompt (Qwen2.5-0.5B-Instruct)

### Location
`scripts/data_process/qa_search_train_merge.py` (line 32-36)

### Raw Prompt Template
```python
prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
```

### After Chat Template Applied (Qwen format)
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of France?
<|im_end|>
<|im_start|>assistant
```

### Key Instructions
1. ✅ Use `<think>` and `</think>` for reasoning
2. ✅ Use `<search>` and `</search>` to query search engine
3. ✅ Receive results in `<information>` and `</information>`
4. ✅ Use `<answer>` and `</answer>` for final answer
5. ✅ Can search multiple times

### Issues Identified ⚠️

**MAJOR ISSUE #1: Wrong Tag Name!**
- **Prompt says:** `<think>` and `</think>`
- **Code expects:** `<thinking>` and `</thinking>` (hierarchical_generation.py doesn't check for `<think>`)
- **Result:** Thinker may generate `<think>` which isn't recognized!

**ISSUE #2: No explicit format enforcement**
- Prompt doesn't require specific order
- Doesn't say "MUST use tags"
- Example only shows final answer format

**ISSUE #3: Vague about "reasoning first"**
- Says "conduct reasoning... first" but doesn't enforce it
- Model might skip thinking and go straight to search/answer

---

## 🔍 SEARCHER Model Prompt (Qwen2.5-1.5B-Instruct)

### Location
`search_r1/search/searcher_server.py` (line 194-211)

### Prompt Template
```python
return f"""<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information.
You can search multiple times if needed (max {self.config.max_turns} searches).

Instructions:
1. First, use <thinking>...</thinking> to reason about what information you need
2. Then use <search>query</search> to search for information
3. You will receive results in <information>...</information>
4. After reviewing the information, either search again or provide <answer>...</answer>
5. Always end with <answer>...</answer> when you have enough information

Provide a concise, direct answer to the query.<|im_end|>
<|im_start|>user
Query: {query}<|im_end|>
<|im_start|>assistant
"""
```

### Full Example
```
<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information.
You can search multiple times if needed (max 3 searches).

Instructions:
1. First, use <thinking>...</thinking> to reason about what information you need
2. Then use <search>query</search> to search for information
3. You will receive results in <information>...</information>
4. After reviewing the information, either search again or provide <answer>...</answer>
5. Always end with <answer>...</answer> when you have enough information

Provide a concise, direct answer to the query.<|im_end|>
<|im_start|>user
Query: What is the capital of France?<|im_end|>
<|im_start|>assistant
```

### Key Instructions
1. ✅ Use `<thinking>` and `</thinking>` (correct tag!)
2. ✅ Use `<search>` and `</search>` to search
3. ✅ Receive results in `<information>` and `</information>`
4. ✅ Use `<answer>` and `</answer>` for final answer
5. ✅ Clear numbered steps
6. ✅ Has system role (better instruction following)

### Strengths ✅
- Correct tag name: `<thinking>` not `<think>`
- Clear step-by-step instructions
- Uses system role for stronger instruction following
- Explicit max turns mentioned
- Numbered list for clarity

---

## ✅ RESOLVED: Tag Consistency

### Current Status (FIXED)

| Component | What It Says | What It Checks |
|-----------|--------------|----------------|
| **Thinker Training Data** | Use `<think>` ✅ | N/A (raw text) |
| **Thinker Code** | Doesn't check for thinking tags | Only checks `<search>` and `<answer>` |
| **Searcher Prompt** | Use `<think>` ✅ | Checks `</think>` ✅ |

### Why This Causes Invalid Actions

Your training data tells Thinker to use `<think>`:
```
You must conduct reasoning inside <think> and </think> first...
```

But the hierarchical_generation.py code doesn't check for `<think>` OR `<thinking>`!

```python
# In hierarchical_generation.py lines 180-218
if '<search>' in response and '</search>' in response:
    # Valid search ✅
elif '<answer>' in response and '</answer>' in response:
    # Valid answer ✅
else:
    # Invalid action ❌ - This catches EVERYTHING else!
```

**Result:** If Thinker generates `<think>`, it's treated as **invalid action**!

---

## 🛠️ Recommended Fixes

### Option 1: Update Thinker Training Data (RECOMMENDED)

**File:** `scripts/data_process/qa_search_train_merge.py`

**Change line 33 from:**
```python
You must conduct reasoning inside <think> and </think> first every time you get new information. \
```

**To:**
```python
You must conduct reasoning inside <thinking> and </thinking> first every time you get new information. \
```

**Then regenerate the training data:**
```bash
python scripts/data_process/qa_search_train_merge.py \
    --input data/hotpotqa_only/raw/ \
    --output data/hotpotqa_only/ \
    --template_type base
```

### Option 2: Make Prompt More Explicit

**Improve the instruction clarity:**

```python
prefix = f"""Answer the given question by following these steps:

1. ALWAYS start with <thinking>your reasoning here</thinking> to analyze the question
2. If you need information, use <search>your query</search>
3. You will receive results in <information>results here</information>
4. After reviewing information, either:
   - Search again with <search>new query</search>
   - Provide final answer with <answer>your answer</answer>

Important:
- You MUST use one of these actions: <thinking>, <search>, or <answer>
- Always close your tags properly
- Keep answers concise, e.g., <answer>Paris</answer>

Question: {question}"""
```

### Option 3: Accept Both `<think>` and `<thinking>` in Code

**File:** `search_r1/llm_agent/hierarchical_generation.py`

**Add after line 218:**
```python
elif ('<thinking>' in response and '</thinking>' in response) or \
     ('<think>' in response and '</think>' in response):
    # Just thinking, continue to next turn
    next_obs_list.append('\nContinue with your action.\n')
    dones.append(False)
    valid_actions.append(1)  # Count as valid
    is_searches.append(0)
```

---

## 📊 Comparison Table

| Feature | Thinker Prompt | Searcher Prompt |
|---------|----------------|-----------------|
| **System Role** | ❌ No (embedded in user message) | ✅ Yes (separate system message) |
| **Tag Name** | ⚠️ `<think>` (inconsistent!) | ✅ `<thinking>` (correct) |
| **Instructions** | ❌ Paragraph format, vague | ✅ Numbered steps, clear |
| **Format Enforcement** | ❌ "You must conduct..." | ✅ "ALWAYS end with..." |
| **Examples** | ✅ Yes (`<answer> Beijing </answer>`) | ⚠️ No explicit example |
| **Max Turns Mentioned** | ✅ "as many times as you want" | ✅ "max 3 searches" |
| **Action Flow** | ❌ Not sequential | ✅ Clear 1→2→3→4 flow |

---

## 🎯 Why You're Seeing Invalid Actions

Based on the prompts, here's what's happening:

### Scenario 1: Thinker generates `<think>` (following training data)
```
<think>I need to find out about Paris</think>
```
**Result:** ❌ Invalid action (no `<search>` or `<answer>` tags)

### Scenario 2: Thinker generates text without tags
```
Let me think about this question...
```
**Result:** ❌ Invalid action (no tags at all)

### Scenario 3: Thinker generates incomplete/malformed tags
```
<search>What is the population of Paris
```
**Result:** ❌ Invalid action (missing closing `</search>`)

### Scenario 4: Thinker generates correct format (after learning)
```
<thinking>I need to find information about Paris</thinking>
<search>What is the capital of France?</search>
```
**Result:** ✅ Valid search (delegated to Searcher)

---

## 🔍 How to Debug

### 1. Check what Thinker actually generates

Add this to `hierarchical_generation.py` after line 160:

```python
for idx, (response, is_active) in enumerate(zip(responses_str, active_mask)):
    if idx < 3:  # Print first 3
        print(f"\n[DEBUG] Sample {idx} full response:")
        print(f"{response}")
        print(f"---")
```

### 2. Check tag detection

Add after line 179:

```python
print(f"[DEBUG] Sample {idx}: Has <search>: {'<search>' in response}, Has </search>: {'</search>' in response}")
print(f"[DEBUG] Sample {idx}: Has <answer>: {'<answer>' in response}, Has </answer>: {'</answer>' in response}")
print(f"[DEBUG] Sample {idx}: Has <think>: {'<think>' in response}, Has <thinking>: {'<thinking>' in response}")
```

### 3. Check extracted content

Add after line 182:

```python
if search_query:
    print(f"[DEBUG] Extracted query: '{search_query}'")
else:
    print(f"[DEBUG] Search query is EMPTY! Response was: {response[:200]}")
```

---

## ✅ Action Items

1. **Immediate Fix:** Regenerate training data with `<thinking>` instead of `<think>`
2. **Code Enhancement:** Make prompt more explicit and structured
3. **Add Debugging:** Print first few samples to see what Thinker generates
4. **Monitor Metrics:** Track `env/ratio_of_valid_action` - should increase to 70%+ after 20 steps

---

## 📁 Files to Review/Modify

1. **Training Data Generation:**
   - `scripts/data_process/qa_search_train_merge.py` (line 32-36)
   - `scripts/data_process/qa_search_test_merge.py` (line 32-36)

2. **Thinker Logic:**
   - `search_r1/llm_agent/hierarchical_generation.py` (lines 179-218)

3. **Searcher Prompt:**
   - `search_r1/search/searcher_server.py` (lines 194-211) ✅ Already correct

---

## 💡 Summary

**The root cause of invalid actions:**
1. Training data says use `<think>` ❌
2. Code doesn't check for `<think>` or `<thinking>` ❌  
3. Code only accepts `<search>` or `<answer>` ✅
4. Searcher correctly uses `<thinking>` ✅

**The fix:**
Update training data to use `<thinking>` consistently, or update code to accept both tag variants, or make prompts more explicit about required format.

Would you like me to create the fixes for you?

