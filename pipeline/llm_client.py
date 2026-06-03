import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

class LLMClient:
    def __init__(self, model_name, model_type, device='cuda',
                 token=None, trust_remote_code=False):
        self.model_type = model_type
        kw = {'token': token} if token else {}
        if trust_remote_code:
            kw['trust_remote_code'] = True
        device_map = device if device == 'auto' else {'': device}

        if model_type == 'gemma3':
            self.processor = AutoProcessor.from_pretrained(model_name, **kw)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device_map,
                attn_implementation='eager', **kw,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map, **kw,)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _gemma3_prompt(self, system, user):
        text = f'{system}\n\n{user}' if system else user
        bos = self.tokenizer.bos_token or '<bos>'
        return f'{bos}<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n'

    @torch.no_grad()
    def call(self, system, user, max_new=512):
        if self.model_type == 'gemma3':
            prompt = self._gemma3_prompt(system, user)
            inputs = self.tokenizer(prompt, return_tensors='pt',
                                    add_special_tokens=False,
                                    return_token_type_ids=False).to(self.model.device)
        else:
            msgs = []
            if system:
                msgs.append({'role': 'system', 'content': system})
            msgs.append({'role': 'user', 'content': user})
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
