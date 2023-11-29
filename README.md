# ntuadlhw3

## run qlora training

``` bash 3-finetune.sh ```

## run public test ppl

``` bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json ```

## run ablation experiments

### zero-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --no_lora --num_samples 150 --no_output ```
### few-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --no_lora --num_samples 150 --no_output --few_shot ```
### lora
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --num_samples 150 --no_output ```
### lora with few-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --num_samples 150 --no_output --few_shot ```
