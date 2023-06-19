from bs4 import BeautifulSoup
import requests
import re


def get_npr_article(url: str, user_agent: str = None) -> str:
    if not user_agent:
        headers = {
            # Random user agent pulled from web
            'User-Agent': 'Mozilla/5.0 (Linux i575 x86_64; en-US) AppleWebKit/601.41 (KHTML, like Gecko) Chrome/47.0.2569.188 Safari/533'
        }
    else:
        headers = {'User-Agent': user_agent}
        
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, 'html5lib')
    
    # Remove Elements 
    for cap in soup.find_all('div', class_='caption'):  # get rid of div captions
        cap.decompose()
    soup.find('p', class_='byline__name').decompose()  # author name
    for embed in soup.find_all('p', id=re.compile('^responsive-embed-')):  # embeds
        embed.decompose()
    for quote in soup.find_all('aside', attrs={'aria-label': 'pullquote'}):  # pull quotes
        quote.decompose()
    for cap in soup.find_all('p', class_='caption'):  # p captions
        cap.decompose()
        
    paragraphs = soup.find_all('p')
    # TODO: This currently returns two sponsor lines in the last part. Either hardcode the last two lines or do not in ['', "Sponsor Message", "Become an NPR sponsor"]

    return ''.join(p.get_text() for p in paragraphs if p != '')  # concatenate paragraphs
    

if __name__ == '__main__':
    example_url = "https://www.npr.org/2023/06/04/1171159008/eric-investigation-voter-data-election-integrity"
    npr_article = get_npr_article(example_url)

    with open("npr_test.txt", 'w') as f:
        f.write(npr_article)
