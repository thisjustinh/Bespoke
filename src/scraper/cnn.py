from bs4 import BeautifulSoup
import requests
import re


def get_cnn_article(url: str, user_agent: str = None) -> str:
    if not user_agent:
        headers = {
            # Random user agent pulled from web
            'User-Agent': 'Mozilla/5.0 (Linux i575 x86_64; en-US) AppleWebKit/601.41 (KHTML, like Gecko) Chrome/47.0.2569.188 Safari/533'
        }
    else:
        headers = {'User-Agent': user_agent}

    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, 'html5lib')
    paragraphs = soup.find_all('p', class_='paragraph')  # get article paragraphs
    # concatenate all lines and return
    # CNN tabs in their paragraphs, except they use spaces, so it's really annoying to remove >:(
    return ''.join(re.sub(r'[\s]{3,}',
                          '',
                          re.sub(r'[\n\t]*', '', p.get_text())) for p in paragraphs if p != '')


if __name__ == '__main__':
    example_url = 'https://www.cnn.com/2023/06/06/tech/ducking-iphone-keyboard-apple/index.html'
    cnn_article = get_cnn_article(example_url)
    print(cnn_article)
    with open('cnn_test.txt', 'w') as f:
        f.write(cnn_article)
        