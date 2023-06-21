import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from app.cnn import get_cnn_article

import time

load_start = time.time()

model_path = "thisjustinh/falcon-7b-cnn-dailymail"
config = LoraConfig.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, model_path)
# model = model.merge_and_unload()
model = model.to("cuda:0")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# tokenizer.pad_token = tokenizer.eos_token


def generate_summary(url):
    article = get_cnn_article(url)
    prompt = f"### Article: {article}\n ###Summary: "
    device = "cuda:0"

    infer_start = time.time()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(input_ids=inputs['input_ids'], 
                             attention_mask=inputs['attention_mask'],
                             eos_token_id=tokenizer.eos_token_id,
                             max_new_tokens=100,
                             do_sample=True,
                             top_k=10,
                             num_return_sequences=1)

    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)

    return summary[0], time.time() - infer_start

