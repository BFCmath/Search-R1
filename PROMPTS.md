# Prompt Formats Reference

This document contains all prompt formats used in the Search-R1 hierarchical framework.

---

## 📚 Table of Contents

1. [Thinker Model Prompts](#thinker-model-prompts)
   - [Search-enabled (Multi-hop QA)](#1-search-enabled-multi-hop-qa)
   - [RAG with Context](#2-rag-with-context)
   - [Simple QA (No Search)](#3-simple-qa-no-search)
2. [Searcher Model Prompts](#searcher-model-prompts)
   - [Searcher Server](#1-searcher-server)
   - [Searcher Agent](#2-searcher-agent)
3. [Tag Reference](#tag-reference)
4. [Example Conversations](#example-conversations)

---

## 🧠 Thinker Model Prompts

The Thinker is the main agent trained with RL. It receives questions and decides when to search or answer.

### 1. Search-enabled (Multi-hop QA)

**Used for:** HotpotQA, NQ, TriviaQA, PopQA, etc.  
**File:** `scripts/data_process/qa_search_train_merge.py`, `qa_search_test_merge.py`, `nq_search.py`

#### Raw Template
```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

#### After Qwen Chat Template
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of France?
<|im_end|>
<|im_start|>assistant
```

#### Key Features
- ✅ Tags: `<think>`, `<search>`, `<information>`, `<answer>`
- ✅ Multi-turn search allowed
- ✅ Must reason with `<think>` after each new information
- ✅ Provides example: `<answer> Beijing </answer>`

---

### 2. RAG with Context

**Used for:** NQ with pre-retrieved context  
**File:** `scripts/data_process/nq_rag.py`

#### Raw Template
```
Answer the given question with some potentially useful context. You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: {question} Context: {context}
```

#### After Qwen Chat Template
```
<|im_start|>user
Answer the given question with some potentially useful context. You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: What is the population of Tokyo? Context: Tokyo is the capital of Japan and one of the most populous cities in the world...
<|im_end|>
<|im_start|>assistant
```

#### Key Features
- ✅ Tags: `<think>`, `<answer>` (no search)
- ✅ Context provided upfront
- ✅ Must evaluate context relevance
- ✅ Simpler format (RAG-style)

---

### 3. Simple QA (No Search)

**Used for:** Basic NQ without search  
**File:** `scripts/data_process/nq.py`

#### Raw Template
```
Answer the given question. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: {question}
```

#### After Qwen Chat Template
```
<|im_start|>user
Answer the given question. You should first have a reasoning process in mind and then provides the answer. Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. Question: What is 2 + 2?
<|im_end|>
<|im_start|>assistant
```

#### Key Features
- ✅ Tags: `<think>`, `<answer>` only
- ✅ No search capability
- ✅ Simplest format
- ✅ For questions answerable without external knowledge

---

## 🔍 Searcher Model Prompts

The Searcher is a frozen model that performs multi-turn search loops when delegated by the Thinker.

### 1. Searcher Server

**Used for:** Hierarchical framework (production)  
**File:** `search_r1/search/searcher_server.py`  
**Model:** Qwen2.5-1.5B-Instruct

#### Full Template
```
<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information.
You can search multiple times if needed (max {max_turns} searches).

Instructions:
1. First, use <think>...</think> to reason about what information you need
2. Then use <search>query</search> to search for information
3. You will receive results in <information>...</information>
4. After reviewing the information, either search again or provide <answer>...</answer>
5. Always end with <answer>...</answer> when you have enough information

Provide a concise, direct answer to the query.<|im_end|>
<|im_start|>user
Query: {query}<|im_end|>
<|im_start|>assistant
```

#### Example (max_turns=3)
```
<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information.
You can search multiple times if needed (max 3 searches).

Instructions:
1. First, use <think>...</think> to reason about what information you need
2. Then use <search>query</search> to search for information
3. You will receive results in <information>...</information>
4. After reviewing the information, either search again or provide <answer>...</answer>
5. Always end with <answer>...</answer> when you have enough information

Provide a concise, direct answer to the query.<|im_end|>
<|im_start|>user
Query: Who created Wallace and Gromit and what other work did they create?<|im_end|>
<|im_start|>assistant
```

#### Key Features
- ✅ System role for better instruction following
- ✅ Numbered instructions for clarity
- ✅ Max turns explicitly stated
- ✅ Clear flow: think → search → information → answer
- ✅ Returns concise summary to Thinker

---

### 2. Searcher Agent

**Used for:** Standalone search agent (testing/development)  
**File:** `search_r1/search/searcher_agent.py`  
**Model:** Qwen2.5-1.5B-Instruct

#### Full Template (Initial)
```
<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information when needed.

Instructions:
1. First, use <think>...</think> to reason about what information you need
2. If you need to search, use <search>query</search> to search for information
3. You will receive results in <information>...</information>
4. After reviewing information, either search again or provide <answer>...</answer>
5. Always end with <answer>...</answer> when you have enough information to answer the query

Important: Focus on gathering factual information to directly answer the query. Be concise but thorough.
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
```

#### Key Features
- ✅ Similar to Searcher Server but slightly different wording
- ✅ Emphasizes concise + thorough balance
- ✅ Optional search ("if you need to search")
- ✅ Can be used standalone without Thinker

---

### 3. Forced Answer Prompt (Searcher)

**Used for:** Final turn when Searcher reaches max turns without answering  
**Files:** `search_r1/search/searcher_agent.py`, `search_r1/search/searcher_server.py`  
**Trigger:** After `max_turns` reached without `<answer>` tag

#### Full Template
```
<|im_start|>system
You have reached the maximum number of search turns. Based on all the information you have gathered so far, you MUST now provide a final answer.

IMPORTANT: You MUST respond with <answer>your answer here</answer>

If you have enough information, provide the best answer you can.
If you don't have enough information, provide <answer>Unable to find sufficient information to answer the query.</answer>

Respond now with <answer>...</answer><|im_end|>
<|im_start|>assistant
```

#### How It Works
```python
# After max_turns reached without answer
def _force_final_answer(current_context, trajectory):
    # 1. Append forced answer prompt to context
    forced_prompt = self._create_forced_answer_prompt()
    final_context = current_context + forced_prompt_tokens
    
    # 2. Generate with greedy decoding (no sampling)
    output = model.generate(
        **final_context,
        do_sample=False,  # ← Deterministic
        max_new_tokens=max_response_length
    )
    
    # 3. Extract answer or use default
    if '<answer>' in output:
        return extract_answer(output)
    else:
        return "Unable to find sufficient information..."
```

#### Key Features
- ✅ **Explicit enforcement:** "MUST respond with <answer>"
- ✅ **Fallback option:** Provides default text if no info found
- ✅ **Greedy decoding:** `do_sample=False` for deterministic output
- ✅ **Double fallback:** If still no tags, returns default message
- ✅ **Logged separately:** "Final Turn (Forced)" in trajectory

#### Example Flow

**Turn 1-3: Normal searching**
```
Turn 1: <think>...</think><search>query 1</search>
Turn 2: <think>...</think><search>query 2</search>
Turn 3: <think>...</think><search>query 3</search>
```

**Max turns reached, force answer:**
```
System: [Appends forced answer prompt]

Model: <answer>Based on the search results, the answer is...</answer>

Result: ✅ Successfully extracted answer
```

**If model still doesn't provide answer:**
```
System: [Appends forced answer prompt]

Model: I think maybe it could be...

Result: ⚠️ No tags, return "Unable to find sufficient information..."
```

#### Why This Approach?

**Problem:** Searcher sometimes keeps searching forever or doesn't provide answer

**Solution:** Explicit prompt forces answer generation in final turn

**Benefits:**
1. ✅ Guarantees Searcher always returns something (never hangs)
2. ✅ Provides useful fallback message if no info found
3. ✅ Uses greedy decoding for more reliable format following
4. ✅ Clearly marked in trajectory for debugging
5. ✅ Graceful degradation (default message if still fails)

---

## 🏷️ Tag Reference

### Tags Used in Prompts

| Tag | Purpose | Used By | Required? |
|-----|---------|---------|-----------|
| `<think>...</think>` | Internal reasoning/thinking | Thinker, Searcher | Recommended |
| `<search>...</search>` | Query to search engine | Thinker, Searcher | When info needed |
| `<information>...</information>` | Search results/context | System (returned) | N/A (system-provided) |
| `<answer>...</answer>` | Final answer | Thinker, Searcher | Required to finish |

### Tag Validation Rules

**For Thinker (in hierarchical mode):**
```python
# Valid actions (hierarchical_generation.py)
if '<search>' in response and '</search>' in response:
    ✅ Valid - delegate to Searcher
elif '<answer>' in response and '</answer>' in response:
    ✅ Valid - provide final answer
else:
    ❌ Invalid - no recognized action
```

**For Searcher:**
```python
# Valid actions (searcher_server.py)
if '</think>' in response:
    ✅ Thinking (continues to next turn)
elif '</search>' in response:
    ✅ Search (execute retrieval)
elif '</answer>' in response:
    ✅ Answer (return to Thinker)
else:
    ❌ Invalid - no recognized action
```

---

## 💬 Example Conversations

### Example 1: Single Search

**Thinker receives:**
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of France?
<|im_end|>
<|im_start|>assistant
```

**Thinker generates:**
```
<think>This is asking about the capital of France. This is common knowledge.</think>
<answer>Paris</answer>
```

**Result:** ✅ Valid answer, done!

---

### Example 2: Multi-turn with Searcher

**Thinker receives:**
```
Question: Who created Wallace and Gromit and what award did they win in 1993?
```

**Thinker Turn 1:**
```
<think>I need to find who created Wallace and Gromit first.</think>
<search>Who created Wallace and Gromit?</search>
```

**↓ Searcher receives query and searches:**

**Searcher Turn 1:**
```
<think>I need to find the creator of Wallace and Gromit.</think>
<search>Wallace and Gromit creator</search>
```

**Searcher receives information:**
```
<information>Nick Park is the British animator who created Wallace and Gromit...</information>
```

**Searcher Turn 2:**
```
<answer>Nick Park created Wallace and Gromit. He is a British animator and filmmaker.</answer>
```

**↓ Thinker receives Searcher's answer:**

**Thinker receives:**
```
<information>Nick Park created Wallace and Gromit. He is a British animator and filmmaker.</information>
```

**Thinker Turn 2:**
```
<think>Now I know Nick Park created it. I need to find what award he won in 1993.</think>
<search>Nick Park award 1993</search>
```

**↓ Searcher receives and processes...**

**Searcher returns:**
```
<answer>Nick Park won the Academy Award for Best Animated Short Film in 1993 for "The Wrong Trousers".</answer>
```

**Thinker receives:**
```
<information>Nick Park won the Academy Award for Best Animated Short Film in 1993 for "The Wrong Trousers".</information>
```

**Thinker Turn 3 (Final):**
```
<think>I now have all the information needed.</think>
<answer>Nick Park created Wallace and Gromit and won the Academy Award for Best Animated Short Film in 1993 for "The Wrong Trousers".</answer>
```

**Result:** ✅ Complete answer with multiple searches!

---

### Example 3: Invalid Action (Common Issue)

**Thinker generates:**
```
Let me think about this... I need to search for information.
```

**Result:** ❌ Invalid action
- No `<think>` tags
- No `<search>` tags
- No `<answer>` tags
- System returns: "My previous action is invalid. If I want to search, I should put the query between <search> and </search>..."

---

## 🔧 Prompt Engineering Tips

### For Better Thinker Performance

1. **Be explicit about tags:** Always emphasize proper tag usage
2. **Provide examples:** Show correct format: `<answer> Paris </answer>`
3. **Enforce structure:** Use "MUST use", "ALWAYS include"
4. **Show consequences:** Explain what happens with invalid format

### For Better Searcher Performance

1. **System role:** Use system message for instruction following
2. **Numbered steps:** Clear 1→2→3→4 flow
3. **Max turns:** State limits explicitly
4. **Concise answers:** Emphasize brevity for Thinker consumption

### Common Issues to Avoid

❌ **Don't:** Mix tag names (`<think>` vs `<thinking>`)  
✅ **Do:** Use consistent tag names across all prompts

❌ **Don't:** Vague instructions like "you may want to..."  
✅ **Do:** Clear instructions like "you MUST use..."

❌ **Don't:** Long, paragraph-style instructions  
✅ **Do:** Numbered, step-by-step instructions

❌ **Don't:** Assume model knows format  
✅ **Do:** Provide explicit examples

---

## 📊 Prompt Comparison

| Feature | Thinker (Search) | Thinker (RAG) | Thinker (Simple) | Searcher |
|---------|------------------|---------------|------------------|----------|
| **Role** | User message | User message | User message | System message |
| **Tags** | 4 tags | 2 tags | 2 tags | 4 tags |
| **Search** | Yes | No | No | Yes |
| **Examples** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Max turns** | Unlimited | 1 | 1 | 3 (configurable) |
| **Format** | Paragraph | Paragraph | Paragraph | Numbered list |
| **Strictness** | Medium | Low | Low | High |

---

## 🎯 Quick Reference

### Thinker Training Data
```
<think>reasoning</think>
<search>query</search>
<information>results</information>
<answer>final answer</answer>
```

### Searcher Service
```
<think>reasoning</think>
<search>query</search>
<information>results</information>
<answer>summary</answer>
```

### What Gets Trained
- ✅ **Thinker tokens:** All `<think>`, `<search>`, `<answer>` tokens
- ❌ **Searcher summaries:** Tokens in `<information>` tags (masked)
- ✅ **Observations:** System messages and error messages

---

## 📁 File Locations

### Data Preparation
- `scripts/data_process/qa_search_train_merge.py` - Training data (search-enabled)
- `scripts/data_process/qa_search_test_merge.py` - Test data (search-enabled)
- `scripts/data_process/nq_search.py` - NQ with search
- `scripts/data_process/nq_rag.py` - NQ with context
- `scripts/data_process/nq.py` - NQ simple

### Runtime Prompts
- `search_r1/search/searcher_server.py` - Searcher server (hierarchical)
- `search_r1/search/searcher_agent.py` - Searcher agent (standalone)
- `search_r1/llm_agent/hierarchical_generation.py` - Thinker logic

---

## ✅ Checklist for New Prompts

When creating new prompts:

- [ ] Use consistent tag names (`<think>`, not `<thinking>`)
- [ ] Include clear examples
- [ ] State max iterations/turns if applicable
- [ ] Use numbered instructions for complex flows
- [ ] Test with small model first
- [ ] Verify tag detection in code
- [ ] Check token masking (for hierarchical mode)
- [ ] Document in this file!

---

**Last Updated:** 2024
**Maintained by:** Search-R1 Team

