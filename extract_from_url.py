# Searches a description for additional text in linked html pages
# Specifically: 1) looks for sentences with keywords "recipe", "steps"
# "cook", "procedure", "preparation", or "method"; 2) locate all URLs
# in that sentence and extract the text
usage = "python extract_from_url.py [path to description file]"

import sys
import re
import nltk.data
import urllib
from io import open
from bs4 import BeautifulSoup

def get_additional_text(url):
  html = urllib.urlopen(url).read()
  soup = BeautifulSoup(html, 'html.parser')
  [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
  all_text = soup.get_text()
  return all_text

def find_url(text):
  search = re.search("(?P<url>https?://[^\s]+)", text)
  if search:
    return search.group("url")
  return None

def extract_from_url(filepath):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  with open(filepath, "a+", encoding="utf-8") as f:
    f.seek(0)
    text = f.read()
    sentences = tokenizer.tokenize(text)
    urls = []
    for sent in sentences:
      lower = sent.lower()
      if "recipe" in lower or \
      "steps" in lower or \
      "cook" in lower or \
      "procedure" in lower or \
      "preparation" in lower:
        url = find_url(sent)
        if url: urls.append(url)

    if len(urls) > 0:
      print("Extracting from URLs in " + filepath)
    for url in urls:
      add = get_additional_text(url)
      f.write(add)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print usage
    sys.exit(-1)
  extract_from_url(sys.argv[1])


