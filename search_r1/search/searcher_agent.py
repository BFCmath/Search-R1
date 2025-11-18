"""
SearcherAgent: A proper agent framework for search tasks

Implements iterative thinking-search-observe loops to answer search queries.
Similar to the Thinker agent but optimized for search and retrieval tasks.
"""

import torch
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests


@dataclass
class SearcherAgentConfig:
    """Configuration for SearcherAgent"""
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_turns: int = 3
    max_response_length: int = 400
    max_obs_length: int = 400
    retrieval_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    device: str = "cuda"
    temperature: float = 0.7
    top_p: float = 0.9


class SearcherAgent:
    """
    SearcherAgent: Implements proper agent reasoning for search tasks

    Uses iterative thinking-search-observe loops to answer queries by:
    1. Thinking about what information is needed
    2. Searching for relevant information
    3. Observing search results
    4. Either searching again or providing a final answer
    """

    def __init__(self, config: SearcherAgentConfig):
        self.config = config

        print(f"\n{'='*80}")
        print(f"🤖 [SEARCHER AGENT] Loading model: {config.model_path}")
        print(f"{'='*80}\n")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ [SEARCHER AGENT] Model loaded successfully!")
        print(f"   Device: {config.device}")
        print(f"   Max turns: {config.max_turns}")
        print(f"   Retrieval URL: {config.retrieval_url}\n")

    @torch.no_grad()
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a search query using agent reasoning loop.

        Args:
            query: The search query to answer

        Returns:
            Dict with 'answer', 'trajectory', 'num_searches', 'success'
        """
        print(f"\n{'='*80}")
        print(f"🔍 [SEARCHER AGENT] Processing query")
        print(f"{'='*80}")
        print(f"Query: {query[:200]}...")
        print(f"{'='*80}\n")

        # Create initial prompt for the agent
        initial_prompt = self._create_initial_prompt(query)

        # Initialize agent state
        current_context = self.tokenizer(
            initial_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.config.device)

        trajectory = [f"Initial Query: {query}"]
        num_searches = 0
        turn = 0

        # Agent reasoning loop
        while turn < self.config.max_turns:
            turn += 1
            print(f"🧠 [Searcher Agent Turn {turn}/{self.config.max_turns}]")

            # Generate agent response
            output = self.model.generate(
                **current_context,
                max_new_tokens=self.config.max_response_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode new tokens only
            new_tokens = output[0][current_context['input_ids'].shape[1]:]
            response_str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Post-process response to stop at action tags
            response_str = self._postprocess_response(response_str)
            trajectory.append(f"Turn {turn}: {response_str[:200]}...")

            print(f"   Response: {response_str[:100]}...")

            # Check if agent provided final answer
            if '<answer>' in response_str and '</answer>' in response_str:
                answer = self._extract_answer(response_str)
                print(f"✅ [Searcher Agent] Found answer after {turn} turns")
                print(f"   Answer: {answer[:100]}...")

                return {
                    'answer': answer,
                    'trajectory': trajectory,
                    'num_searches': num_searches,
                    'success': True
                }

            # Check if agent wants to search
            elif '<search>' in response_str and '</search>' in response_str:
                search_query = self._extract_search_query(response_str)
                if search_query:
                    print(f"   🔎 Searching: {search_query[:80]}...")

                    # Execute search
                    search_results = self._execute_search(search_query)
                    num_searches += 1

                    # Create observation
                    observation = f'\n\n<information>{search_results}</information>\n\n'
                    trajectory.append(f"Search {num_searches}: {search_query}")
                    trajectory.append(f"Results: {search_results[:200]}...")

                    # Update context with response + observation
                    response_tokens = self.tokenizer.encode(
                        response_str, add_special_tokens=False, return_tensors='pt'
                    ).to(self.config.device)
                    observation_tokens = self.tokenizer.encode(
                        observation, add_special_tokens=False, return_tensors='pt'
                    ).to(self.config.device)

                    # Concatenate to context
                    current_context['input_ids'] = torch.cat([
                        current_context['input_ids'],
                        response_tokens,
                        observation_tokens
                    ], dim=1)

                    # Truncate if too long (keep recent context)
                    if current_context['input_ids'].shape[1] > 2048:
                        current_context['input_ids'] = current_context['input_ids'][:, -2048:]

                    # Update attention mask
                    current_context['attention_mask'] = torch.ones_like(
                        current_context['input_ids']
                    )
                else:
                    print(f"   ❌ Invalid search query")
                    break
            else:
                # Invalid action
                print(f"   ❌ Invalid action (no search or answer)")
                break

        # No answer found within max turns
        print(f"⚠️ [Searcher Agent] No answer after {self.config.max_turns} turns")

        return {
            'answer': "Unable to find sufficient information to answer the query.",
            'trajectory': trajectory,
            'num_searches': num_searches,
            'success': False
        }

    def _create_initial_prompt(self, query: str) -> str:
        """Create initial prompt for SearcherAgent"""
        return f"""<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information when needed.

Instructions:
1. First, use <thinking>...</thinking> to reason about what information you need
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
"""

    def _postprocess_response(self, response: str) -> str:
        """Post-process response to stop at action tags"""
        # Stop at the first action tag
        if '</thinking>' in response:
            response = response.split('</thinking>')[0] + '</thinking>'
        elif '</search>' in response:
            response = response.split('</search>')[0] + '</search>'
        elif '</answer>' in response:
            response = response.split('</answer>')[0] + '</answer>'

        return response

    def _extract_search_query(self, text: str) -> str:
        """Extract search query from <search>...</search>"""
        match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_answer(self, text: str) -> str:
        """Extract answer from <answer>...</answer>"""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return everything after <answer>
        if '<answer>' in text:
            return text.split('<answer>')[1].strip()
        return text.strip()

    def _execute_search(self, query: str) -> str:
        """Execute search via retrieval API"""
        try:
            payload = {
                "queries": [query],
                "topk": self.config.topk,
                "return_scores": True
            }
            response = requests.post(
                self.config.retrieval_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()['result'][0]

                # Format results
                formatted = ""
                for idx, item in enumerate(results):
                    doc = item['document']
                    content = doc['contents']
                    title = content.split("\n")[0].replace('"', '').strip()
                    text_content = "\n".join(content.split("\n")[1:])
                    formatted += f"Doc {idx+1} (Title: {title}): {text_content}\n\n"

                return formatted.strip()
            else:
                print(f"   ❌ Search API returned {response.status_code}")
                return "Search failed due to API error."

        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return "Search failed due to connection error."
