# Task 5 Results: Test the Fine-Tuned Agent

## Purpose

Task 5 compares the original SmolLM2 model with the LoRA adapter trained in
Task 4. It tests a normal taco-restaurant question and an unrelated question
to check topic relevance and character consistency.

## Execution

The task was run twice from the repository root with the CPU virtual
environment:

```bash
cd /home/irom/fine-tuning-llms
source .venv/bin/activate
python task_5_test_agent.py
python task_5_test_agent.py
```

Both runs loaded the base model and the local adapter from `lora_adapter/`.
The model ran successfully on CPU; a GPU was not required.

## Test prompts

Normal customer prompt:

```text
What's your best seller?
```

Off-topic prompt:

```text
What's the capital of France?
```

The script checks whether a response contains at least two TacoBot-related
keywords. This is a lightweight heuristic, not a full semantic evaluation.

## Results

Because generation uses sampling, the output and scores varied between runs.

| Run | Base score | Fine-tuned score | Base normal test | Fine-tuned normal test | Base off-topic test | Fine-tuned off-topic test |
|---|---:|---:|---|---|---|---|
| `20260926T143324Z` | 0/2 | 0/2 | No | No | No | No |
| `20260926T143340Z` | 2/2 | 1/2 | Yes | Yes | Yes | No |

The first run classified both models as off-topic for both checks. The second
run classified both models as on-topic for the normal prompt, while only the
base model passed the off-topic character check.

## Preserved reports

Each execution creates a unique timestamped report instead of overwriting a
previous result:

```text
results/task5/run_20260926T143324Z/report.json
results/task5/run_20260926T143340Z/report.json
```

Each report contains the prompts, generated responses, matched keywords,
adapter status, and comparison scores.

## Interpretation

The two runs do not demonstrate a consistent improvement from the short
50-step fine-tuning run. The fine-tuned model sometimes remained relevant to
the normal TacoBot prompt, but it did not consistently stay in character for
the unrelated prompt.

This variability is expected because the script uses sampled generation with a
small base model and a small training dataset. More training data, more
training steps, deterministic evaluation, and a stronger evaluator would be
needed for a reliable behavioral comparison.

## Evaluation limitation

The `is_on_topic` check counts keyword matches. It does not fully determine
whether the response is helpful, correctly formatted, faithful to the TacoBot
role, or appropriately refuses an unrelated request. A response can mention a
taco keyword while still answering the off-topic question incorrectly.

For stronger evaluation, inspect the saved responses and additionally check:

1. Whether the answer is relevant to the requested scenario.
2. Whether it remains a TacoBot restaurant response.
3. Whether it avoids confidently answering unrelated questions when that is
   against the intended behavior.
4. Whether it follows the required JSON response schema, if that requirement
   is enabled for the agent.

## Output artifact

Successful execution writes:

```text
markers/task5_complete.txt
```

The marker contains:

```text
TEST_AGENT_COMPLETE
```

