from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

learning_rate = 3e-4
max_seq_length = 512
hidden_size = 256
intermediate_size = 1024
num_hidden_layers = 8
num_attention_heads = 8
num_train_epochs = 4
gradient_accumulation_steps = 128
per_device_train_batch_size = 20
effective_batch_size = gradient_accumulation_steps * per_device_train_batch_size
warmup_steps = 30
lr_scheduler_type = "cosine"

dataset_path = dict(
    path="HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    cache_dir="/mnt/hp-ssd/datasets",
    revision='5b89d1ea9319fe101b3cbdacd89a903aca1d6052'
)

print(f'batch size = {effective_batch_size:,}')
print('dataset', dataset_path)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.model_max_length = max_seq_length

model_config = LlamaConfig(
	vocab_size=tokenizer.vocab_size + 1,
	max_position_embeddings=tokenizer.model_max_length,
	hidden_size=hidden_size,
	intermediate_size=intermediate_size,
	num_hidden_layers=num_hidden_layers,
	num_attention_heads=num_attention_heads,
	tie_word_embeddings=True,
	pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
)
model = LlamaForCausalLM(model_config)
print(f'model parameters = {model.num_parameters():,}')

dataset = load_dataset(**dataset_path)
num_samples = dataset['train'].num_rows
print(f'training samples = {num_samples:,}')
max_steps = (num_samples // effective_batch_size) * num_train_epochs
print(f'max training steps = {max_steps:,}')

dataset = load_dataset(**dataset_path, streaming=True)
train_dataset = dataset['train']

remove_columns = set(train_dataset.features.keys()) - set(['text'])
train_dataset = train_dataset.remove_columns(remove_columns)
train_dataset = train_dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
    batched=True
)
eval_dataset = None
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=False,
    save_strategy="epoch",
    evaluation_strategy="no",
    #num_train_epochs=num_train_epochs, # overwritten by max_steps when data is streaming
    max_steps=max_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    warmup_steps=warmup_steps, 
    lr_scheduler_type=lr_scheduler_type,
    learning_rate=learning_rate,
    save_total_limit=1,
    logging_steps=1,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
trainer.save_model("./output/finished")
