## Baby LLaMA
```
hardware = 2 x RTX3060
batch size = 2,560 (1.5K to 4K according to *Cramming BERT* [1])
model parameters = 16,585,216 (check out the original BabyLlama repo [2])
dataset = HuggingFaceFW/fineweb-edu (46 GiB disk space)
training tokens = sample-10BT (should be over the ~20 tokens/parameter per *Chinchilla law* [2])
training samples = 9,672,101
training steps = 3,778
time per step = 150 sec/step
time estimated = 156:42:50 (about 6.5 days)
```

## Checkpoint-2150 Example Output
```
<s> Baby Llama is a 40-year old-state healthy baby or teen that is known to be a long-hoven diet and a healthier diet for young children and adults in high-risk areas.
What is 325-67 years ago? There is a clear link between these 30-day meals and other health problems and how to avoid this complication by helping adults with 36-hour period of all-to-go-effects.
What is 338-287-039-5
```

## Reference
1. Cramming BERT (https://arxiv.org/pdf/2212.14034)
2. Original BabyLlama size configurations (https://github.com/timinar/BabyLlama/tree/main/config)
3. Chinchilla Law (https://arxiv.org/abs/2203.15556)
