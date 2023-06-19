import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
# from scraper.forbes import get_forbes_article

# Make sure you have the above installed and the following:
# bitsandbytes einops accelerate wandb

model_path = "./finetune/falcon-7b-cnn-dailymail/checkpoint-500"

config = LoraConfig.from_pretrained(model_path)

print(f"Loading model {config.base_model_name_or_path}...")

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    # model_path
    # trust_remote_code=True
)

print("Model loaded")
print("Pushing to hub...")

model.push_to_hub('falcon-7b-cnn-dailymail')

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# article = get_forbes_article("https://www.forbes.com/sites/kenrickcai/2023/06/04/stable-diffusion-emad-mostaque-stability-ai-exaggeration/?sh=5fad3cc175c5")
with open('./scraper/cnn_test.txt', 'r') as f:
    article = f.read()

print(article)

prompt = f"### Article: {article}\n###Summary: "
device = "cuda:0"

model = model.to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
