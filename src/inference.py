import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, PeftModel
# from scraper.forbes import get_forbes_article

# Make sure you have the above installed and the following:
# bitsandbytes einops accelerate wandb

# model_path = "./finetune/falcon-7b-cnn-dailymail/checkpoint-500"
# model_path = "thisjustinh/falcon-7b-cnn-dailymail"
model_path = "./finetune/checkpoint-500"

config = LoraConfig.from_pretrained(model_path)

print(config)
print(f"Loading model {config.base_model_name_or_path}...")

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0}
)

print("Model loaded, converting to PEFT")

model = PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()

# print("Pushing to hub...")
# model.push_to_hub('falcon-7b-cnn-dailymail')

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# article = get_forbes_article("https://www.forbes.com/sites/kenrickcai/2023/06/04/stable-diffusion-emad-mostaque-stability-ai-exaggeration/?sh=5fad3cc175c5")
with open('./scraper/cnn_test.txt', 'r') as f:
    article = f.read()

# print(article)

prompt = f"### Article: {article}\n###Summary: "
device = "cuda:0"

model = model.to(device)
model.eval()

print("Beginning Inference")

inputs = tokenizer(prompt, return_tensors='pt').to(device)
print(inputs)
outputs = model.generate(inputs['input_ids'], max_new_tokens=200)

summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

print("Inference done!")

print(summary)
with open('cnn_inference.txt', 'w') as f:
    f.write(summary[0])

