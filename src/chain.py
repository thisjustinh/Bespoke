from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from scraper.forbes import get_forbes_article

# Set API Key
load_dotenv(find_dotenv())

falcon_instruct = HuggingFaceHub(repo_id='google/flan-t5-xl')

forbes_article = get_forbes_article('https://www.forbes.com/sites/kenrickcai/2023/06/04/stable-diffusion-emad-mostaque-stability-ai-exaggeration/?sh=5fad3cc175c5')

# TODO: This currently doesn't work LOL.
prompt = f"The article is as follows: {forbes_article}. The summary is "
completion = falcon_instruct(prompt)
print(completion)
