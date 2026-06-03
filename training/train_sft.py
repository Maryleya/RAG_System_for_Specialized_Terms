import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import hf_hub_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lora-r', type=int, default=64)
    p.add_argument('--lora-alpha', type=int, default=128)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--per-device-bs', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--trust-remote-code', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f'Model:  {args.model}\nData:   {args.data}\nOutput: {args.output}')
    print(f'Params: epochs={args.epochs}, lr={args.lr}, r={args.lora_r}, '
          f'alpha={args.lora_alpha}, max_len={args.max_length}, '
          f'bs={args.per_device_bs}, grad_accum={args.grad_accum}\n')

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, padding_side='right')
    if not tokenizer.chat_template:
        with open(hf_hub_download(args.model, 'config.json')) as f:
            tokenizer.chat_template = json.load(f).get('chat_template_jinja')
        print(f'chat_template set from config.json: {bool(tokenizer.chat_template)}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_prompt(system, user):
        msgs = []
        if system: 
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': user})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f'Loading data from {args.data}...')
    items = json.load(open(args.data))
    ds = Dataset.from_list([
        {'prompt':     build_prompt(it['system'], it['user']),
         'completion': it['completion'] + tokenizer.eos_token}
        for it in items
    ])
    print(f'dataset: {len(ds)} items')

    print(f'Loading base model ({args.model}, bf16)...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
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
        max_length=args.max_length,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tokenizer)
    print('Starting SFT training...')
    trainer.train()

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f'Saved SFT LoRA adapter to {args.output}')

if __name__ == '__main__':
    main()
