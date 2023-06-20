import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

# Make sure you have the above installed and the following:
# bitsandbytes einops accelerate wandb

# Some hyperparameters:
output_dir = 'falcon-7b-cnn-dailymail'
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
per_device_train_batch_size = 5
gradient_accumulation_steps = 5
optim = 'paged_adamw_32bit'
save_steps = 125
logging_steps = 10
learning_rate = 2e-5
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = 'constant'
max_sequence_length = 1024

dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
model_name = 'ybelkada/falcon-7b-sharded-bf16'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=[
        'query_key_value',
        'dense',
        'dense_h_to_4h',
        'dense_4h_to_h',
    ]
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=True
)


def prompt_formatting_func(example):
    example['text'] = f"### Article: {example['article']}\n ### Summary: {example['highlights']} <|endoftext|>"
    return example

print("### Starting Prompt Dataset Creation")
prompt_dataset = dataset.map(prompt_formatting_func)
print("### Finished Prompt Dataset Creation")

# print(prompt_dataset[0])
# import sys
# sys.exit(0)

trainer = SFTTrainer(
    model=model,
    train_dataset=prompt_dataset,
    # train_dataset=dataset.with_format('torch'),
    dataset_text_field='text',
    # formatting_func=prompt_formatting_func,
    peft_config=peft_config,
    max_seq_length=max_sequence_length,
    tokenizer=tokenizer,
    args=training_args
)

for name, module in trainer.model.named_modules():
    if 'norm' in name:
        module = module.to(torch.float32)

# Some sanity checks
# print(prompt_formatting_func(dataset[1000]))
# print(dataset.with_format('torch')[0])
# print(prompt_dataset[0])

trainer.train()
trainer.push_to_hub()
