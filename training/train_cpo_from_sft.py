import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import Dataset
from trl import CPOTrainer, CPOConfig
from huggingface_hub import hf_hub_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--sft-lora', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lora-r', type=int, default=64)
    p.add_argument('--lora-alpha', type=int, default=128)
    p.add_argument('--per-device-bs', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--trust-remote-code', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f'Model:    {args.model}\nSFT LoRA: {args.sft_lora}\nOutput:   {args.output}')
    print(f'Params: epochs={args.epochs}, lr={args.lr}, beta={args.beta}, r={args.lora_r}, alpha={args.lora_alpha}\n')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, padding_side='left')
    if not tokenizer.chat_template:
        with open(hf_hub_download(args.model, 'config.json')) as f:
            tokenizer.chat_template = json.load(f).get('chat_template_jinja')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_prompt(text):
        msgs = [{'role': 'user', 'content':
            f'Translate the following text to Russian. Provide only the translation, '
            f'without any explanations or comments.\n\n{text}'}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    pool = json.load(open(args.data))
    ds = Dataset.from_list([
        {'prompt':   build_prompt(it['source_en']),
         'chosen':   it['chosen'],
         'rejected': it['rejected']}
        for it in pool if it.get('chosen') and it.get('rejected')
    ])
    print(f'dataset: {len(ds)} items')

    print(f'Loading base model ({args.model})...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=args.trust_remote_code,
    )
    print(f'Merging SFT LoRA from {args.sft_lora}...')
    model = PeftModel.from_pretrained(model, args.sft_lora)
    model = model.merge_and_unload()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    cpo_cfg = CPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=20,
        save_strategy='steps',
        save_steps=600,
        save_total_limit=1,
        report_to='none',
        beta=args.beta,
        max_prompt_length=512,
        max_length=1024,
        loss_type='sigmoid',
    )

    trainer = CPOTrainer(model=model, args=cpo_cfg, train_dataset=ds, processing_class=tokenizer)
    print('Starting CPO training (ALMA-R style)...')
    trainer.train()

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f'Saved CPO LoRA adapter to {args.output}')

if __name__ == '__main__':
    main()
