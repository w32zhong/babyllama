## Baby LLaMA
```
hardware = 2 x RTX3060
batch size = 2,560 (1.5K to 4K according to *Cramming BERT* [1])
model parameters = 16,585,216 (check out the original BabyLlama repo [2])
dataset = HuggingFaceFW/fineweb-edu
training tokens = sample-10BT (should be over the ~20 tokens/parameter *Chinchilla law* [2])
training samples = 9,672,101
training steps = 3,778
time per step = 167.00s/it
time estimated = 175:12:42 (about 7 days)
```

## Reference
1. Cramming BERT (https://arxiv.org/pdf/2212.14034)
2. Original BabyLlama (https://github.com/timinar/BabyLlama/tree/main/config)
3. Chinchilla Law (https://arxiv.org/abs/2203.15556)
