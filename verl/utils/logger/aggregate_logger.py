# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""
import numbers
from typing import Dict


def concat_dict_to_str(dict: Dict, step):
    output = [f'step:{step}']
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f'{k}:{v:.3f}')
    output_str = ' - '.join(output)
    return output_str


def format_metrics_compact(metrics: Dict) -> str:
    """Format metrics in a compact, readable way for console output."""
    # Prioritize important metrics
    priority_keys = [
        'critic/rewards/mean', 'critic/score/mean',
        'critic/kl', 'critic/kl_coeff',
        'actor/approx_kl', 'actor/loss',
        'env/finish_ratio', 'env/number_of_actions/mean',
        'timing_s/step', 'timing_s/gen'
    ]
    
    output_parts = []
    shown_keys = set()
    
    # Show priority metrics first
    for key in priority_keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, numbers.Number):
                short_key = key.split('/')[-1]  # Get last part of key
                output_parts.append(f'{short_key}={val:.3f}')
                shown_keys.add(key)
    
    # Show remaining metrics (limited to prevent clutter)
    remaining = [(k, v) for k, v in metrics.items() 
                 if k not in shown_keys and isinstance(v, numbers.Number)]
    if len(remaining) > 0:
        output_parts.append('|')
        for k, v in remaining[:5]:  # Limit to 5 additional metrics
            short_key = k.split('/')[-1]
            output_parts.append(f'{short_key}={v:.3f}')
    
    return ' '.join(output_parts)


class LocalLogger:

    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        if print_to_console:
            print('Using LocalLogger is deprecated. The constructor API will change ')

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            # Use compact formatting for better readability
            metrics_str = format_metrics_compact(data)
            print(f'[Step {step}] {metrics_str}', flush=True)