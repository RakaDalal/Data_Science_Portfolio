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
		r.append(w)

r=list()
z=z[1:]
for w in z:
	add(w,r)
for v in r:
        print v
c=0;
p=[]
for v in r:
	url=v
	c=c+1
	print(c)
	response=urllib2.urlopen(url)
	p.append(response.read())


i=0
		      
h=[]
for i in range(len(r)):
        h.append([])  
	print(i)
	soup=BeautifulSoup(p[i])
	g=soup.find_all('li')
	for elem in g:
		v=elem.get('id')
		if(v!=None):
		      if(v.startswith('cite')):
			      h[i].append(elem.text)


	
