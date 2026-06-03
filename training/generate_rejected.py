import argparse
import json
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "trust_remote_code": False,
    },
    "hy-mt2": {
        "model_name": "tencent/Hy-MT2-7B",
        "trust_remote_code": True,
    },
}


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", choices=["qwen", "hy-mt2"], required=True)

    p.add_argument("--pool-in", required=True)
    p.add_argument("--pool-out", required=True)

    p.add_argument("--cache")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)

    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    if args.cache is None:
        args.cache = f"rejected_cache_{args.model}.json"

    print(
        f"Model:      {cfg['model_name']}\n"
        f"Pool in:    {args.pool_in}\n"
        f"Pool out:   {args.pool_out}\n"
        f"Cache:      {args.cache}\n"
        f"Batch size: {args.batch_size}\n"
    )

    print("Loading pool...")
    pool = json.load(open(args.pool_in))
    print(f"pool: {len(pool)} items")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=cfg["trust_remote_code"], padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg["trust_remote_code"],
    )

    model.eval()

    def build_prompt(text):
        messages = [
            {
                "role": "user",
                "content": (
                    "Translate the following text to Russian. "
                    "Provide only the translation, without any "
                    "explanations or comments.\n\n"
                    f"{text}"
                ),
            }
        ]

        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def translate_batch(texts):
        prompts = [build_prompt(text) for text in texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]

        return [tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip() for i in range(len(texts))]

    unique = sorted({item["source_en"] for item in pool})
    print(f"unique source_en: {len(unique)}")

    if os.path.exists(args.cache):
        rejected_cache = json.load(open(args.cache))
    else:
        rejected_cache = {}
    print(f"cache resumed: {len(rejected_cache)} entries")

    pending = [text for text in unique if text not in rejected_cache]
    print(f"pending: {len(pending)}")

    for i in tqdm(range(0, len(pending), args.batch_size), desc=f"{args.model} translate"):
        batch = pending[i : i + args.batch_size]

        outputs = translate_batch(batch)

        for src, out in zip(batch, outputs):
            rejected_cache[src] = out

        if (i // args.batch_size) % 50 == 0:
            json.dump(rejected_cache, open(args.cache, "w"), ensure_ascii=False)

    json.dump(rejected_cache, open(args.cache, "w"), ensure_ascii=False)
    print(f"cached {len(rejected_cache)} translations")

    for item in pool:
        item["rejected"] = rejected_cache.get(item["source_en"], "")
        item["chosen"] = item["source_ru"]

    before = len(pool)

    pool = [item for item in pool if item["rejected"] and item["chosen"]]
    print(f"pool: {before} -> {len(pool)} (dropped empty rejected/chosen)")

    json.dump(pool, open(args.pool_out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.pool_out}")

    ex = pool[0]
    print(
        f"\nsource_en: {ex['source_en'][:120]}\n"
        f"chosen:    {ex['chosen'][:120]}\n"
        f"rejected:  {ex['rejected'][:120]}"
    )

if __name__ == "__main__":
    main()