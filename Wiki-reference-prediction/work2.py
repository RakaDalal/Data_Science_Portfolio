import nltk
import urllib2
from bs4 import BeautifulSoup
url="https://en.wikipedia.org/wiki/Book:Information_retrieval"
response=urllib2.urlopen(url)
chapters=response.read()
z=list()
soup = BeautifulSoup(chapters)
for w in soup.find_all('a'):
	z.append(w.get('href'))

def add(w,r):
	if w.startswith('/wiki'):
		w='http://en.wikipedia.org'+w
		#print("2")
		r.append(w)
r=list()
z=z[1:]
for w in z:
	add(w,r)
for v in r:
        print v
