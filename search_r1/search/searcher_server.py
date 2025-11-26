"""
Searcher Agent Server

A separate service that runs the Searcher model (Qwen2.5-1.5B).
Receives queries from Thinker, performs search-observe loops, returns summaries.

Usage:
    python search_r1/search/searcher_server.py --model_path Qwen/Qwen2.5-1.5B-Instruct --port 8001
"""

import torch
import re
import argparse
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class SearcherConfig:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_turns: int = 3,
        max_response_length: int = 400,
        max_obs_length: int = 400,
        retrieval_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.max_turns = max_turns
        self.max_response_length = max_response_length
        self.max_obs_length = max_obs_length
        self.retrieval_url = retrieval_url
        self.topk = topk
        self.device = device


class SearcherAgent:
    """
    Searcher Agent: Performs search-observe loops and returns summaries
    """
    def __init__(self, config: SearcherConfig):
        self.config = config
        
        print(f"\n{'='*80}")
        print(f"🤖 [SEARCHER] Loading model: {config.model_path}")
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
        
        print(f"✅ [SEARCHER] Model loaded successfully!")
        print(f"   Device: {config.device}")
        print(f"   Max turns: {config.max_turns}")
        print(f"   Retrieval URL: {config.retrieval_url}\n")
    
    @torch.no_grad()
    def execute_search(self, thinker_query: str) -> Dict:
        """
        Execute search loop for a Thinker's query.
        
        Args:
            thinker_query: Query from the Thinker agent
            
        Returns:
            Dict with 'summary', 'trajectory', 'num_searches'
        """
        print(f"\n{'='*80}")
        print(f"🔍 [SEARCHER] Received query from Thinker")
        print(f"{'='*80}")
        print(f"FULL QUERY:")
        print(thinker_query)
        print(f"{'='*80}\n")
        
        # Create prompt for Searcher
        searcher_prompt = self._create_searcher_prompt(thinker_query)
        
        print(f"\n{'─'*80}")
        print(f"📝 [SEARCHER] Initial Prompt (Full):")
        print(f"{'─'*80}")
        print(searcher_prompt)
        print(f"{'─'*80}\n")
        
        # Tokenize
        current_context = self.tokenizer(
            searcher_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.config.device)
        
        trajectory = [f"Initial Query: {thinker_query}"]
        num_searches = 0
        
        # Search-observe loop
        for turn in range(self.config.max_turns):
            print(f"  🔄 [Searcher Turn {turn + 1}/{self.config.max_turns}]")
            
            # Generate response
            output = self.model.generate(
                **current_context,
                max_new_tokens=self.config.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode only new tokens
            new_tokens = output[0][current_context['input_ids'].shape[1]:]
            response_str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Post-process to stop at tags
            if '</think>' in response_str:
                response_str = response_str.split('</think>')[0] + '</think>'
            elif '</search>' in response_str:
                response_str = response_str.split('</search>')[0] + '</search>'
            elif '</answer>' in response_str:
                response_str = response_str.split('</answer>')[0] + '</answer>'
            
            trajectory.append(f"Turn {turn + 1}: {response_str}")
            
            print(f"\n{'─'*80}")
            print(f"💬 [SEARCHER] Turn {turn + 1} FULL Response:")
            print(f"{'─'*80}")
            print(response_str)
            print(f"{'─'*80}\n")
            
            # Check if answer provided
            if '</answer>' in response_str:
                answer = self._extract_answer(response_str)
                print(f"  ✅ [Searcher] Found answer after {turn + 1} turns")
                print(f"     Answer: {answer[:100]}...")
                return {
                    'summary': answer,
                    'trajectory': trajectory,
                    'num_searches': num_searches,
                    'success': True
                }
            
            # If search, execute it
            if '</search>' in response_str:
                search_query = self._extract_search_query(response_str)
                if search_query:
                    print(f"\n{'─'*80}")
                    print(f"🔎 [SEARCHER] Search Query {num_searches + 1}:")
                    print(f"{'─'*80}")
                    print(search_query)
                    print(f"{'─'*80}\n")
                    
                    search_results = self._do_search(search_query)
                    num_searches += 1
                    
                    print(f"\n{'─'*80}")
                    print(f"📚 [SEARCHER] Search Results {num_searches} (FULL):")
                    print(f"{'─'*80}")
                    print(search_results)
                    print(f"{'─'*80}\n")
                    
                    observation = f'\n\n<information>{search_results}</information>\n\n'
                    trajectory.append(f"Search {num_searches}: {search_query}")
                    trajectory.append(f"Results: {search_results}")
                    
                    # Update context - append tokens directly to avoid losing information
                    response_tokens = self.tokenizer.encode(response_str, add_special_tokens=False, return_tensors='pt').to(self.config.device)
                    observation_tokens = self.tokenizer.encode(observation, add_special_tokens=False, return_tensors='pt').to(self.config.device)
                    
                    # Concatenate tokens
                    current_context['input_ids'] = torch.cat([
                        current_context['input_ids'],
                        response_tokens,
                        observation_tokens
                    ], dim=1)
                    
                    # Truncate from left if too long to keep recent context
                    if current_context['input_ids'].shape[1] > 2048:
                        current_context['input_ids'] = current_context['input_ids'][:, -2048:]
                    
                    # Update attention mask
                    current_context['attention_mask'] = torch.ones_like(current_context['input_ids'])
                else:
                    print(f"     ⚠️ Invalid search query")
                    break
            else:
                print(f"     ⚠️ Invalid action (no search or answer)")
                break
        
        # Force final answer generation after max turns
        print(f"\n{'='*80}")
        print(f"⚠️ [SEARCHER] Max turns reached, forcing final answer generation")
        print(f"{'='*80}\n")
        
        forced_answer = self._force_final_answer(current_context, trajectory)
        
        print(f"\n{'='*80}")
        print(f"✅ [SEARCHER] FINAL SUMMARY:")
        print(f"{'='*80}")
        print(forced_answer)
        print(f"{'='*80}\n")
        
        return {
            'summary': forced_answer,
            'trajectory': trajectory,
            'num_searches': num_searches,
            'success': False if forced_answer.startswith("Unable to") else True
        }
    
    def _create_searcher_prompt(self, query: str) -> str:
        """Create prompt for Searcher agent"""
        return f"""<|im_start|>system
You are a search assistant. Your task is to answer the query by searching for information.
You can search multiple times if needed (max {self.config.max_turns} searches).

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
"""
    
    def _create_forced_answer_prompt(self) -> str:
        """Create prompt that forces answer generation in final turn"""
        return f"""\n\n<|im_start|>system
You have reached the maximum number of search turns. Based on all the information you have gathered so far, you MUST now provide a final answer.

IMPORTANT: You MUST respond with <answer>your answer here</answer>

If you have enough information, provide the best answer you can.
If you don't have enough information, provide <answer>Unable to find sufficient information to answer the query.</answer>

Respond now with <answer>...</answer><|im_end|>
<|im_start|>assistant
"""
    
    def _force_final_answer(self, current_context: Dict, trajectory: List[str]) -> str:
        """
        Force the model to generate a final answer when max turns reached.
        
        Args:
            current_context: Current context with all history
            trajectory: Conversation trajectory for logging
            
        Returns:
            Extracted answer string
        """
        print(f"\n{'─'*80}")
        print(f"🎯 [SEARCHER] Forced Answer Prompt:")
        print(f"{'─'*80}")
        forced_prompt = self._create_forced_answer_prompt()
        print(forced_prompt)
        print(f"{'─'*80}\n")
        
        # Add forced answer prompt to context
        forced_prompt_tokens = self.tokenizer.encode(
            forced_prompt, add_special_tokens=False, return_tensors='pt'
        ).to(self.config.device)
        
        # Append to context
        final_context = {
            'input_ids': torch.cat([
                current_context['input_ids'],
                forced_prompt_tokens
            ], dim=1),
        }
        
        # Truncate if needed
        if final_context['input_ids'].shape[1] > 2048:
            final_context['input_ids'] = final_context['input_ids'][:, -2048:]
        
        final_context['attention_mask'] = torch.ones_like(final_context['input_ids'])
        
        # Generate final answer
        output = self.model.generate(
            **final_context,
            max_new_tokens=self.config.max_response_length,
            do_sample=False,  # Greedy for final answer
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only new tokens
        new_tokens = output[0][final_context['input_ids'].shape[1]:]
        response_str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Post-process
        if '</answer>' in response_str:
            response_str = response_str.split('</answer>')[0] + '</answer>'
        
        trajectory.append(f"Final Turn (Forced): {response_str}")
        
        print(f"\n{'─'*80}")
        print(f"📝 [SEARCHER] Forced Response (FULL):")
        print(f"{'─'*80}")
        print(response_str)
        print(f"{'─'*80}\n")
        
        # Extract answer
        if '<answer>' in response_str and '</answer>' in response_str:
            answer = self._extract_answer(response_str)
            print(f"✅ [SEARCHER] Successfully extracted answer")
            return answer
        else:
            print(f"⚠️ [SEARCHER] No answer tags even after forcing, using default")
            return "Unable to find sufficient information to answer the query."
    
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
    
    def _do_search(self, query: str) -> str:
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
            results = response.json()['result'][0]
            
            # Format results
            formatted = ""
            for idx, item in enumerate(results):
                doc = item['document']
                content = doc['contents']
                title = content.split("\n")[0].replace('"', '').strip()
                text = "\n".join(content.split("\n")[1:])
                formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
            
            return formatted.strip()
        except Exception as e:
            print(f"     ❌ Search error: {e}")
            return "Search failed due to connection error."


#####################################
# FastAPI Server
#####################################

class SearchQuery(BaseModel):
    query: str
    max_turns: Optional[int] = None


class SearchResponse(BaseModel):
    summary: str
    trajectory: List[str]
    num_searches: int
    success: bool


app = FastAPI(title="Searcher Agent Server")


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchQuery):
    """
    Endpoint for Thinker to delegate search queries.
    
    Input:
    {
        "query": "What is the capital of France?",
        "max_turns": 3  // optional
    }
    
    Output:
    {
        "summary": "The capital of France is Paris.",
        "trajectory": ["Turn 1: ...", "Turn 2: ..."],
        "num_searches": 2,
        "success": true
    }
    """
    # Temporarily override max_turns if provided
    original_max_turns = searcher_agent.config.max_turns
    if request.max_turns:
        searcher_agent.config.max_turns = request.max_turns
    
    try:
        result = searcher_agent.execute_search(request.query)
        return SearchResponse(**result)
    finally:
        # Restore original max_turns
        searcher_agent.config.max_turns = original_max_turns


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": config.model_path,
        "max_turns": config.max_turns
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the Searcher Agent Server"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path to Searcher model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run server on"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Maximum search turns for Searcher"
    )
    parser.add_argument(
        "--retrieval_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="URL of retrieval server"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of documents to retrieve per search"
    )
    
    args = parser.parse_args()
    
    # Create global config and agent
    config = SearcherConfig(
        model_path=args.model_path,
        max_turns=args.max_turns,
        retrieval_url=args.retrieval_url,
        topk=args.topk,
    )
    
    searcher_agent = SearcherAgent(config)
    
    print(f"\n{'='*80}")
    print(f"🚀 [SEARCHER SERVER] Starting on http://0.0.0.0:{args.port}")
    print(f"{'='*80}\n")
    
    # Launch server
    uvicorn.run(app, host="0.0.0.0", port=args.port)

