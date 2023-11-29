from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str, few_shot: bool) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # p = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。"
    p = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供文言文與白話文之間的翻譯。"
    p += f"文言文又稱古文。現代文又稱白話文。"
    if few_shot:
        p += f"文言文：雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。 現代文：雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。"
        p += f"文言文：後未旬，果見囚執。 現代文：沒過十天，鮑泉果然被拘捕。"
    p += f"USER: {instruction} ASSISTANT:"
    return p


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

