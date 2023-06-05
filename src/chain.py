from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from scraper.forbes import get_forbes_article

# Set API Key
load_dotenv(find_dotenv())

falcon_instruct = HuggingFaceHub(repo_id='tiiuae/falcon-7b')

forbes_article = get_forbes_article('https://www.forbes.com/sites/kenrickcai/2023/06/04/stable-diffusion-emad-mostaque-stability-ai-exaggeration/?sh=5fad3cc175c5')

# TODO: This currently doesn't work LOL
prompt = f"The article is as follows:\n\n{forbes_article}. The summary is "
print(prompt)
completion = falcon_instruct(prompt)
print(completion)
