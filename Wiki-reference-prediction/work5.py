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
	if w.startswith('//'):
		w='http:'+w
		print("1")
		r.append(w)
	elif w.startswith('/wiki'):
		w='http://en.wikipedia.org'+w
		print("2")
		r.append(w)
	elif w.startswith('http'):
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
		      
citation=[]
f=open("F:\\references_inf.txt",'w')
for i in range(len(r)):
        citation.append(0)
	print(i)
	f.write('\n'+r[i].encode('utf-8')+"             " + '\n'*5);
	soup=BeautifulSoup(p[i])
	g=soup.find_all('li')
	for elem in g:
		v=elem.get('id')
		if(v!=None):
		      if(v.startswith('cite')):
			      citation[i]=citation[i]+1

f.close()
citation2=[]
for w in citation:
    if(w!=0):
       citation2.append(w)
print(sum(citation2)/len(citation2))

import matplotlib.pyplot as plt
n,bins,patches = plt.hist(citation2,10,facecolor='g',alpha=0.75)
plt.xlabel('No of References')
plt.ylabel('No of Pages')
plt.title('Histogram of references')
plt.axis([0, 60, 0, 30])
plt.grid(True)
plt.show()
'''
f=open("F:\\References2_inf.txt",'w')
for i in range(len(r)):
	print(i)
	f.write('\n'+r[i].encode('utf-8')+"             " + '\n'*5);
	soup=BeautifulSoup(p[i])
	g=soup.find_all('li')
	b=[]
	for w in g:
		b.append(w.get('id'))
	g2=soup.find_all('p')
	for el in g2:
		k=[]
		for w in el.find_all('a'):
			if(w!=None):
			     if((w.get('href')).startswith('#cite')):
			           k.append(w.get('href'))
		for ww in k:
			for ss in b:
                               if(ss!=None):
				      if(ss in ww):
					    f.write('\n'+ss+'\n'*2)
					    f.write('\n'*2 + "REFERENCE"+'\n'*2)
			                    f.write((g[b.index(ss)].text).encode('utf-8'))
			                    f.write('\n'*2 + "CONTEXT"+'\n'*2)
			                    f.write(el.text.encode('utf-8')+'\n'*2)

f.close()			                    

'''

	
