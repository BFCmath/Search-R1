"""
Hierarchical Agent System: Thinker-Searcher Framework

Architecture:
- Thinker (0.5B): High-level reasoning, delegates search queries  
- Searcher (1.5B): Separate service, performs search-observe loops, returns summaries

Only the Thinker is trained with RL (searcher tokens are masked).
"""

import torch
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import requests


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical agent system"""
    # Thinker config
    thinker_max_turns: int = 2
    thinker_max_response_length: int = 300
    
    # Searcher config (service)
    searcher_url: str = "http://127.0.0.1:8001/search"
    searcher_max_turns: int = 3
    searcher_max_response_length: int = 400  # Max length for Searcher summary
    searcher_max_obs_length: int = 400       # Max obs length for Searcher
    
    # Shared config
    max_start_length: int = 1024
    max_prompt_length: int = 2048
    num_gpus: int = 1
    
    # Retrieval config (not used directly, Searcher handles it)
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3


class SearcherClient:
    """
    Client to communicate with Searcher service
    """
    def __init__(self, config: HierarchicalConfig):
        self.config = config
    
    def execute_search(self, thinker_query: str) -> str:
        """
        Call Searcher service to execute search loop.
        
        Args:
            thinker_query: Query from the Thinker agent
            
        Returns:
            Summary answer from Searcher
        """
        try:
            payload = {
                "query": thinker_query,
                "max_turns": self.config.searcher_max_turns
            }
            
            response = requests.post(
                self.config.searcher_url,
                json=payload,
                timeout=120  # Searcher may take time
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['summary']
            else:
                print(f"[WARNING] Searcher service returned {response.status_code}")
                return "Searcher service error - unable to process query."
                
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to Searcher service at {self.config.searcher_url}")
            print(f"        Make sure to run: bash launch_searcher.sh")
            return "Searcher service not available."
        except Exception as e:
            print(f"[ERROR] Searcher request failed: {e}")
            return "Searcher service error."


class HierarchicalGenerationManager:
    """
    Manages hierarchical agent system with Thinker and Searcher
    """
    def __init__(
        self,
        thinker_tokenizer,
        thinker_rollout_wg,
        config: HierarchicalConfig,
        is_validation: bool = False,
    ):
        self.thinker_tokenizer = thinker_tokenizer
        self.thinker_rollout_wg = thinker_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        # Initialize Searcher client (calls separate service)
        self.searcher_client = SearcherClient(config)
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=thinker_tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.searcher_max_obs_length,  # Max obs length for Searcher summaries
            max_start_length=config.max_start_length
        ))
    
    def run_hierarchical_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Run hierarchical generation loop with Thinker and Searcher
        
        Returns:
            final_output: DataProto with generation results
            Only Thinker tokens are kept for RL training
        """
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        
        rollings = gen_batch
        
        print(f"\n{'='*80}")
        print(f"🧠 [HIERARCHICAL] Starting Thinker-Searcher Framework")
        print(f"{'='*80}")
        print(f"Thinker: {self.config.thinker_max_turns} turns × {batch_size} samples")
        print(f"Searcher: {self.config.searcher_max_turns} turns per query")
        print(f"{'='*80}\n")
        
        # Log initial prompts
        print(f"\n{'─'*80}")
        print(f"📝 [THINKER] Initial Prompts (FULL):")
        print(f"{'─'*80}")
        for idx in range(batch_size):
            initial_prompt = self.thinker_tokenizer.decode(
                gen_batch.batch['input_ids'][idx], 
                skip_special_tokens=False
            )
            print(f"\n[Sample {idx}] Initial Prompt:")
            print(initial_prompt)
            print(f"{'-'*40}")
        print(f"{'─'*80}\n")
        
        # Main Thinker loop
        for thinker_turn in range(self.config.thinker_max_turns):
            if not active_mask.sum():
                break
                
            print(f"🧠 [Thinker Turn {thinker_turn + 1}/{self.config.thinker_max_turns}] Active: {active_mask.sum()}/{batch_size}")
            
            # Cut to effective length before generation
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate Thinker response
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active)
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )
            
            # Log Thinker responses
            print(f"\n{'─'*80}")
            print(f"💬 [THINKER] Turn {thinker_turn + 1} - All Active Sample Responses:")
            print(f"{'─'*80}")
            for idx, (response, is_active) in enumerate(zip(responses_str, active_mask)):
                if is_active:
                    print(f"\n[Sample {idx}] FULL Response:")
                    print(response)
                    print(f"{'-'*40}")
            print(f"{'─'*80}\n")
            
            # Process Thinker actions
            next_obs_list = []
            dones = []
            valid_actions = []
            is_searches = []
            
            for idx, (response, is_active) in enumerate(zip(responses_str, active_mask)):
                if not is_active:
                    next_obs_list.append('')
                    dones.append(True)
                    valid_actions.append(0)
                    is_searches.append(0)
                    continue
                
                # Check if Thinker wants to search
                if '<search>' in response and '</search>' in response:
                    # Extract search query
                    search_query = self._extract_tag_content(response, 'search')
                    if search_query:
                        print(f"\n{'─'*80}")
                        print(f"  🔍 [THINKER Sample {idx}] Delegating to Searcher")
                        print(f"{'─'*80}")
                        print(f"  FULL Search Query:")
                        print(search_query)
                        print(f"{'─'*80}\n")
                        
                        # Delegate to Searcher service via client
                        searcher_summary = self.searcher_client.execute_search(search_query)
                        
                        print(f"\n{'─'*80}")
                        print(f"  ✅ [THINKER Sample {idx}] Received from Searcher:")
                        print(f"{'─'*80}")
                        print(f"  FULL Searcher Summary:")
                        print(searcher_summary)
                        print(f"{'─'*80}\n")
                        
                        # Return Searcher's summary as information to Thinker
                        observation = f'\n\n<information>{searcher_summary}</information>\n\n'
                        next_obs_list.append(observation)
                        dones.append(False)
                        valid_actions.append(1)
                        is_searches.append(1)
                    else:
                        # Invalid search
                        next_obs_list.append(self._get_invalid_action_msg())
                        dones.append(False)
                        valid_actions.append(0)
                        is_searches.append(0)
                
                elif '<answer>' in response and '</answer>' in response:
                    # Thinker provided final answer
                    answer_content = self._extract_tag_content(response, 'answer')
                    print(f"\n{'─'*80}")
                    print(f"  ✅ [THINKER Sample {idx}] Provided FINAL ANSWER:")
                    print(f"{'─'*80}")
                    print(answer_content)
                    print(f"{'─'*80}\n")
                    next_obs_list.append('')
                    dones.append(True)
                    valid_actions.append(1)
                    is_searches.append(0)
                else:
                    # Invalid action
                    print(f"\n  ❌ [THINKER Sample {idx}] Invalid action - no proper tags\n")
                    next_obs_list.append(self._get_invalid_action_msg())
                    dones.append(False)
                    valid_actions.append(0)
                    is_searches.append(0)
            
            # Update states
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_actions, dtype=torch.int)
            valid_search_stats += torch.tensor(is_searches, dtype=torch.int)
            
            # Log observations being fed back to Thinker
            print(f"\n{'─'*80}")
            print(f"📥 [THINKER] Turn {thinker_turn + 1} - Observations Fed Back:")
            print(f"{'─'*80}")
            for idx, obs in enumerate(next_obs_list):
                if obs:
                    print(f"\n[Sample {idx}] Observation:")
                    print(obs)
                    print(f"{'-'*40}")
            print(f"{'─'*80}\n")
            
            # Update rolling context
            next_obs_ids = self._process_next_obs(next_obs_list)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )
        
        # Final Thinker response if still active
        if active_mask.sum():
            print(f"\n{'='*80}")
            print(f"🧠 [THINKER FINAL TURN] Forcing answer from active samples: {active_mask.sum()}")
            print(f"{'='*80}\n")
            
            # Cut to effective length before final generation
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )
            
            # Log final responses
            print(f"\n{'─'*80}")
            print(f"💬 [THINKER FINAL] All Active Sample Final Responses:")
            print(f"{'─'*80}")
            for idx, (response, is_active) in enumerate(zip(responses_str, active_mask)):
                if is_active:
                    print(f"\n[Sample {idx}] FULL Final Response:")
                    print(response)
                    print(f"{'-'*40}")
            print(f"{'─'*80}\n")
            
            original_right_side = self._update_right_side(
                original_right_side, responses_ids
            )
        
        # Prepare metadata
        meta_info = {
            'turns_stats': turns_stats.tolist(),
            'active_mask': active_mask.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'valid_search_stats': valid_search_stats.tolist(),
        }
        
        print(f"\n{'='*80}")
        print(f"✅ [HIERARCHICAL] Complete")
        print(f"{'='*80}\n")
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        """Extract content from XML-style tags"""
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _get_invalid_action_msg(self) -> str:
        """Get invalid action message"""
        return ('\nMy previous action is invalid. '
                'If I want to search, I should put the query between <search> and </search>. '
                'If I want to give the final answer, I should put the answer between <answer> and </answer>. '
                'Let me try again.\n')
    
    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at appropriate tags"""
        responses_str = self.thinker_tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        responses_str = [
            resp.split('</search>')[0] + '</search>' if '</search>' in resp
            else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp
            else resp
            for resp in responses_str
        ]
        
        responses = self.thinker_tokenizer(
            responses_str,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']
        
        return responses, responses_str
    
    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process observations"""
        next_obs_ids = self.thinker_tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.tensor_fn.config.max_obs_length:
            print(f"[WARNING] Observation too long: {next_obs_ids.shape[1]} > {self.tensor_fn.config.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.tensor_fn.config.max_obs_length]
        
        return next_obs_ids
    
    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.thinker_tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info
    
    def _update_rolling_state(self, rollings, cur_responses, next_obs_ids):
        """Update rolling state with new data"""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings
    
    def _update_right_side(self, right_side, cur_responses, next_obs_ids=None):
        """Update right side tracking"""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        
        # Keep within max length
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len]
        }
    
    def _generate_with_gpu_padding(self, active_batch):
        """Generate with GPU padding for multi-GPU"""
        num_gpus = self.config.num_gpus
        
        # Convert to long dtype
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        
        if num_gpus <= 1:
            return self.thinker_rollout_wg.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.thinker_rollout_wg.generate_sequences(active_batch)
        
        # Pad to GPU count
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Convert padded batch to long dtype
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        padded_output = self.thinker_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
        
        padded_output.batch = trimmed_batch
        
        return padded_output
    
    def _compose_final_output(self, left_side, right_side, meta_info):
        """Compose final output"""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
