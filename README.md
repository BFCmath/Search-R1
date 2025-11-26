# Train GRPO for Delegate using Search R1 method

## Methodology
Using same method as Search R1 to train delegate system

The idea is to train the Orchestrator, and treat the Worker as a searcher that return search results. (apply the same methodology as Search R1, we train Orchestrator but ignore the worker output)

## Results
The worker fail to follow the framework (which basically a React pipeline so that the worker can iteratively answer small question by searching the database)

I think figure out a way to Lora SFT the worker first so they can follow the framework is important to make this work.

However, the result is also extremely slow, and SLM (3B) seem cannot handle long context well. 
