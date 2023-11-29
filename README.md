# ntuadlhw3

## run qlora training

``` bash 3-finetune.sh ```

## run public test w/o ppl

``` bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json ```

## run public test ppl

``` bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --do_ppl ```

## run ablation experiments

### zero-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --no_lora --num_samples 150 --no_output --do_ppl ```
### few-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --no_lora --num_samples 150 --no_output --few_shot --do_ppl ```
### lora
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --num_samples 150 --no_output --do_ppl ```
### lora with few-shot
``` time bash run.sh ./Taiwan-LLM-7B-v2.0-chat/ ./adapter_checkpoint/ ./data/public_test.json ./output/public_test.json --num_samples 150 --no_output --few_shot --do_ppl ```
