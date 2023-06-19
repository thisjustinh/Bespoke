from bs4 import BeautifulSoup
import requests
import re


def get_forbes_article(url: str, user_agent: str = None) -> str:
    if not user_agent:
        headers = {
            # Random user agent pulled from web
            'User-Agent': 'Mozilla/5.0 (Linux i575 x86_64; en-US) AppleWebKit/601.41 (KHTML, like Gecko) Chrome/47.0.2569.188 Safari/533'
        }
    else:
        headers = {'User-Agent': user_agent}

    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, 'html5lib')
    paragraphs = soup.find_all('p')  # get article paragraphs
    # concatenate all lines and return 
    return ''.join(re.sub(r'[\n\t]*', '', p.get_text()) for p in paragraphs if p != '')


if __name__ == '__main__':
    example_url = 'https://www.forbes.com/sites/kenrickcai/2023/06/04/stable-diffusion-emad-mostaque-stability-ai-exaggeration/?sh=5fad3cc175c5'
    forbes_article = get_forbes_article(example_url)

    with open('forbes_test.txt', 'w') as f:
        f.write(forbes_article)
