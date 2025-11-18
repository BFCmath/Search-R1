# Enhanced Logging - Full Input/Output Tracking

This document describes the comprehensive logging system for both **Thinker** and **Searcher** agents in the hierarchical framework. All logs print **FULL** content without truncation for complete visibility.

---

## Overview

The logging system provides detailed visibility into:
1. **Thinker Agent**: All prompts, responses, actions, and observations
2. **Searcher Agent**: All prompts, responses, search queries, and results
3. **Complete Flow**: Every interaction between agents with full context

---

## Thinker Agent Logging

### Location
`search_r1/llm_agent/hierarchical_generation.py`

### What's Logged

#### 1. Initial Prompts (Start of Generation)
```
─────────────────────────────────────────────────────────────────────────────────
📝 [THINKER] Initial Prompts (FULL):
─────────────────────────────────────────────────────────────────────────────────

[Sample 0] Initial Prompt:
<full prompt text here>
----------------------------------------

[Sample 1] Initial Prompt:
<full prompt text here>
----------------------------------------
...
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Complete initial prompt for every sample in the batch, including system messages and user queries.

#### 2. Turn Responses (Every Turn)
```
─────────────────────────────────────────────────────────────────────────────────
💬 [THINKER] Turn 1 - All Active Sample Responses:
─────────────────────────────────────────────────────────────────────────────────

[Sample 0] FULL Response:
<think>Complete thinking text</think>
<search>Complete search query</search>
----------------------------------------

[Sample 2] FULL Response:
<think>Complete thinking text</think>
<answer>Final answer</answer>
----------------------------------------
...
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Complete response from Thinker model for each active sample, including all `<think>`, `<search>`, or `<answer>` tags.

#### 3. Search Delegation
```
─────────────────────────────────────────────────────────────────────────────────
  🔍 [THINKER Sample 0] Delegating to Searcher
─────────────────────────────────────────────────────────────────────────────────
  FULL Search Query:
Complete search query text sent to Searcher
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Full search query being delegated to the Searcher service.

#### 4. Searcher Responses Received
```
─────────────────────────────────────────────────────────────────────────────────
  ✅ [THINKER Sample 0] Received from Searcher:
─────────────────────────────────────────────────────────────────────────────────
  FULL Searcher Summary:
Complete summary text returned by Searcher agent
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Full summary/answer returned by the Searcher agent.

#### 5. Final Answers
```
─────────────────────────────────────────────────────────────────────────────────
  ✅ [THINKER Sample 0] Provided FINAL ANSWER:
─────────────────────────────────────────────────────────────────────────────────
Complete final answer text
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Final answer when Thinker provides `<answer>` tag.

#### 6. Invalid Actions
```
  ❌ [THINKER Sample 1] Invalid action - no proper tags
```

**Prints**: When Thinker generates response without proper `<think>`, `<search>`, or `<answer>` tags.

#### 7. Observations Fed Back
```
─────────────────────────────────────────────────────────────────────────────────
📥 [THINKER] Turn 1 - Observations Fed Back:
─────────────────────────────────────────────────────────────────────────────────

[Sample 0] Observation:

<information>Complete information text from Searcher</information>

----------------------------------------
...
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Full observation text being appended to Thinker's context after each action.

#### 8. Final Turn (Forced)
```
═════════════════════════════════════════════════════════════════════════════════
🧠 [THINKER FINAL TURN] Forcing answer from active samples: 2
═════════════════════════════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────────────────────────
💬 [THINKER FINAL] All Active Sample Final Responses:
─────────────────────────────────────────────────────────────────────────────────

[Sample 1] FULL Final Response:
<think>Final thinking</think>
<answer>Forced answer</answer>
----------------------------------------
...
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Final forced responses when max turns are reached.

---

## Searcher Agent Logging

### Location
`search_r1/search/searcher_server.py`

### What's Logged

#### 1. Query Received
```
═════════════════════════════════════════════════════════════════════════════════
🔍 [SEARCHER] Received query from Thinker
═════════════════════════════════════════════════════════════════════════════════
FULL QUERY:
Complete query text from Thinker
═════════════════════════════════════════════════════════════════════════════════
```

**Prints**: Complete query received from the Thinker agent.

#### 2. Initial Prompt
```
─────────────────────────────────────────────────────────────────────────────────
📝 [SEARCHER] Initial Prompt (Full):
─────────────────────────────────────────────────────────────────────────────────
<|im_start|>system
You are a search assistant...
<|im_end|>
<|im_start|>user
Query: [complete query]
<|im_end|>
<|im_start|>assistant

─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Complete initial prompt sent to the Searcher model.

#### 3. Turn Responses
```
─────────────────────────────────────────────────────────────────────────────────
💬 [SEARCHER] Turn 1 FULL Response:
─────────────────────────────────────────────────────────────────────────────────
<think>Complete thinking text</think>
<search>Complete search query</search>
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Complete response from Searcher model each turn.

#### 4. Search Queries
```
─────────────────────────────────────────────────────────────────────────────────
🔎 [SEARCHER] Search Query 1:
─────────────────────────────────────────────────────────────────────────────────
Complete search query sent to retrieval API
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Full search query being sent to the retrieval service.

#### 5. Search Results
```
─────────────────────────────────────────────────────────────────────────────────
📚 [SEARCHER] Search Results 1 (FULL):
─────────────────────────────────────────────────────────────────────────────────
Doc 1 (Title: Title 1): Complete document text...

Doc 2 (Title: Title 2): Complete document text...

Doc 3 (Title: Title 3): Complete document text...
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Complete search results returned by the retrieval API.

#### 6. Forced Answer Prompt (When Max Turns Reached)
```
═════════════════════════════════════════════════════════════════════════════════
⚠️ [SEARCHER] Max turns reached, forcing final answer generation
═════════════════════════════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────────────────────────
🎯 [SEARCHER] Forced Answer Prompt:
─────────────────────────────────────────────────────────────────────────────────

<|im_start|>system
You have reached the maximum number of search turns...
IMPORTANT: You MUST respond with <answer>your answer here</answer>
...
<|im_end|>
<|im_start|>assistant

─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: The forced answer prompt appended to context.

#### 7. Forced Response
```
─────────────────────────────────────────────────────────────────────────────────
📝 [SEARCHER] Forced Response (FULL):
─────────────────────────────────────────────────────────────────────────────────
<answer>Complete forced answer text</answer>
─────────────────────────────────────────────────────────────────────────────────
```

**Prints**: Full response after forcing answer generation.

#### 8. Final Summary
```
═════════════════════════════════════════════════════════════════════════════════
✅ [SEARCHER] FINAL SUMMARY:
═════════════════════════════════════════════════════════════════════════════════
Complete final summary/answer text
═════════════════════════════════════════════════════════════════════════════════
```

**Prints**: Final summary being returned to the Thinker.

---

## Log Structure

### Symbols Used
- `═════` (double line): Major sections/boundaries
- `─────` (single line): Sub-sections/content boundaries
- `-----` (dashes): Individual sample separators
- `🧠`: Thinker agent
- `🔍`: Search action
- `💬`: Model response
- `📝`: Prompt
- `📥`: Observation received
- `📚`: Search results
- `🎯`: Forced action
- `✅`: Success/completion
- `❌`: Error/invalid action
- `⚠️`: Warning

### Log Levels

**High-level (═ borders)**:
- Framework start/end
- Major phase transitions
- Final summaries

**Mid-level (─ borders)**:
- Turn-by-turn responses
- Prompts and observations
- Search queries and results

**Low-level (- separators)**:
- Individual samples in batch
- Content separators

---

## Usage Examples

### Example 1: Tracking a Single Sample Flow

```
[Initial Prompt]
📝 [THINKER] Initial Prompts (FULL):
[Sample 0] Initial Prompt:
Answer the given question. You must conduct reasoning inside <think>...
Question: What is the capital of France?

↓

[Turn 1 Response]
💬 [THINKER] Turn 1 - All Active Sample Responses:
[Sample 0] FULL Response:
<think>I need to search for information about France's capital</think>
<search>capital of France</search>

↓

[Search Delegation]
🔍 [THINKER Sample 0] Delegating to Searcher
FULL Search Query:
capital of France

↓

[SEARCHER logs its entire process...]
🔍 [SEARCHER] Received query from Thinker
...
✅ [SEARCHER] FINAL SUMMARY:
The capital of France is Paris.

↓

[Observation Fed Back]
📥 [THINKER] Turn 1 - Observations Fed Back:
[Sample 0] Observation:

<information>The capital of France is Paris.</information>

↓

[Turn 2 Response]
💬 [THINKER] Turn 2 - All Active Sample Responses:
[Sample 0] FULL Response:
<think>The capital is Paris</think>
<answer>Paris</answer>

↓

[Final Answer]
✅ [THINKER Sample 0] Provided FINAL ANSWER:
Paris
```

### Example 2: Debugging Invalid Actions

When you see:
```
❌ [THINKER Sample 3] Invalid action - no proper tags
```

Look at the preceding response to see what the model generated:
```
💬 [THINKER] Turn 1 - All Active Sample Responses:
[Sample 3] FULL Response:
The answer is probably...
```

This shows the model didn't use proper `<think>`, `<search>`, or `<answer>` tags.

---

## Benefits

### 1. Complete Visibility
- **No truncation**: Every log shows full content
- **Full context**: See exactly what the model sees and generates
- **Debugging**: Easy to identify where things go wrong

### 2. Training Analysis
- **Prompt Engineering**: See if prompts are working as intended
- **Model Behavior**: Track how models learn over time
- **Error Patterns**: Identify common failure modes

### 3. Performance Tracking
- **Search Quality**: See what queries are being made
- **Answer Quality**: Review final answers
- **Iteration Count**: Track how many turns are needed

---

## Configuration

All logging is **always enabled** and prints to stdout. To save logs:

```bash
# Save to file
python your_script.py 2>&1 | tee full_logs.txt

# Save with timestamps
python your_script.py 2>&1 | while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line"; done | tee timestamped_logs.txt
```

---

## Notes

1. **Performance**: Logging adds minimal overhead since it only prints (no file I/O in code)
2. **Volume**: Logs can be large with big batches; use `grep` or filtering to focus on specific samples
3. **Line wrapping**: Long text may wrap in terminal; use wider terminal or redirect to file
4. **Special tokens**: Logs show special tokens like `<|im_start|>` for debugging purposes

---

## Filtering Logs

### Show only Thinker responses:
```bash
grep -A 10 "💬 \[THINKER\]" full_logs.txt
```

### Show only Searcher responses:
```bash
grep -A 10 "💬 \[SEARCHER\]" full_logs.txt
```

### Show only search queries:
```bash
grep -A 5 "🔎 \[SEARCHER\] Search Query" full_logs.txt
```

### Show only final answers:
```bash
grep -A 3 "✅.*FINAL ANSWER" full_logs.txt
```

### Show only errors:
```bash
grep "❌" full_logs.txt
```

---

## Summary

The enhanced logging system provides **complete transparency** into the hierarchical Thinker-Searcher framework:

✅ **Full input/output tracking** - No truncation anywhere  
✅ **Clear visual structure** - Easy to follow the flow  
✅ **Detailed agent behavior** - See every decision and action  
✅ **Debugging-friendly** - Quickly identify issues  
✅ **Analysis-ready** - Track training progress and model behavior  

Use these logs to understand, debug, and improve your hierarchical agent system!

