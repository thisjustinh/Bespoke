import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, PeftModel
# from scraper.forbes import get_forbes_article
from scraper.cnn import get_cnn_article
import time

# Make sure you have the above installed and the following:
# bitsandbytes einops accelerate wandb

load_start = time.time()

# model_path = "./finetune/falcon-7b-cnn-dailymail/checkpoint-500"
model_path = "thisjustinh/falcon-7b-cnn-dailymail"
# model_path = "./finetune/checkpoint-500"

config = LoraConfig.from_pretrained(model_path)

# print(config)
print(f"Loading model {config.base_model_name_or_path}...")

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

print("Model loaded, converting to PEFT")

model = PeftModel.from_pretrained(model, model_path)
# model = model.merge_and_unload()


tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

article = get_cnn_article("https://www.cnn.com/2023/06/20/europe/andrew-tate-charges-trial-intl-gbr/index.html")
# with open('./scraper/forbes_test.txt', 'r') as f:
#     article = f.read()

# print(article)

prompt = f"### Article: {article}\n###Summary: "
device = "cuda:0"

model = model.to(device)
model.eval()

print("Beginning Inference")
infer_start = time.time()
inputs = tokenizer(prompt, return_tensors='pt').to(device)
# print(inputs)
# print(tokenizer("<|endoftext|>"))
outputs = model.generate(input_ids=inputs['input_ids'], 
                         attention_mask=inputs['attention_mask'],
                         eos_token_id=11,
                         pad_token_id=11,
                         bos_token_id=1,
                         max_new_tokens=80)
# outputs = model.generate(**inputs)

summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)

print("Inference done!")
print(f"Model load took {infer_start - load_start} seconds. Inference took {time.time() - infer_start} seconds")

# print(summary)
with open('cnn_inference.txt', 'w') as f:
    f.write(summary[0])

