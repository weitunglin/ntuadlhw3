import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def perplexity(
    model, tokenizer, data, max_length=2048, few_shot=False
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"], few_shot) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out = model(input_ids, attention_mask=attn_mask)
            out_logits = out.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Taiwan-LLM-7B-v2.0-chat",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to output path."
    )
    parser.add_argument(
        "--no_lora",
        action="store_true"
    )
    parser.add_argument(
        "--no_output",
        action="store_true"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None
    )
    parser.add_argument(
        "--few_shot",
        action="store_true"
    )
    parser.add_argument(
        "--do_ppl",
        action="store_true"
    )

    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.no_lora:
        # Load LoRA
        model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    #data = data[:10]

    if isinstance(args.num_samples, int):
        data = data[:args.num_samples]

    model.eval()
    if args.do_ppl:
        ppl = perplexity(model, tokenizer, data, few_shot=args.few_shot)
        print("Mean perplexity:", ppl["mean_perplexity"])

    if not args.no_output:
        gen_config = GenerationConfig(
            num_beams=2,
            do_sample=False,
            max_new_tokens=64
        )
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, generation_config=gen_config)

        output_text = []
        output_id = []
        for i in tqdm(data):
            with torch.no_grad():
                p = get_prompt(i["instruction"], args.few_shot)
                x = pipe(p)
                generated_text = x[0]['generated_text'][len(p)+1:]
                if '答案：' in generated_text:
                    generated_text = generated_text.replace('答案：', '')
                output_text.append(generated_text)
                output_id.append(i["id"])

        result = pd.DataFrame(data={'id':output_id,'output':output_text})
        result.to_json(args.output_path, orient="records")

