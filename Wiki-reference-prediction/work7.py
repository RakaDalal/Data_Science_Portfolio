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

q=[[]]*len(r)
i=0
for i in range(len(r)):
	print(i)
	soup=BeautifulSoup(p[i])
	for w in soup.find_all('a'):
		q[i].append(w.get('href'))	  

r2=[]
rr2=[]
for i in range(len(r)):
        rr2.append([]) 
	print(i)
	soup=BeautifulSoup(p[i])
	g=soup.find_all('li')
	for elem in g:
		v=elem.get('id')
		if(v!=None):
		      if(v.startswith('cite')):
			      rr2[i].append(elem.text.encode('utf-8'))
                              r2.append(elem.text.encode('utf-8'))

r2=list(set(r2))
count=[]
for w in r2:
   count.append(0)
   for i in range(len(r)):
        if w in rr2[i]:
            count[r2.index(w)]=count[r2.index(w)]+1

import matplotlib.pyplot as plt
n,bins,patches = plt.hist(count,5,facecolor='g',alpha=0.75)
plt.xlabel('No of wikipages')
plt.ylabel('No of Refernces')
plt.title('Histogram of references')
plt.axis([0, 10, 0, 600])
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

	
