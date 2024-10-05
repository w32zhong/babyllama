import torch
from transformers import AutoTokenizer
from transformers import TextStreamer, LlamaForCausalLM

import logging
logging.disable()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

streamer = TextStreamer(tokenizer)

model = LlamaForCausalLM.from_pretrained("./output/checkpoint-2300/")
model.eval()
model.to('cuda')

prompt = 'Baby Llama is a '
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to(next(model.parameters()).device)

with torch.no_grad():
    model.generate(**inputs, streamer=streamer, max_length=128, do_sample=True)
