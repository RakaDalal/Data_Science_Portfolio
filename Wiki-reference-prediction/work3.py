import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
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
		w='http://en.wikipedia.org/w/index.php?title='+w[6:]+'&action=history'
		print("2")
		r.append(w)
r=list()
z=z[1:]
for w in z:
	add(w,r)
for v in r:
        print v
setpages=[[]]
q=[]
i=0
for i in range(len(r)):
	print(i)
        rr=0
        q.append([])
        response=urllib2.urlopen(r[i])
        p=response.read()
	soup=BeautifulSoup(p)
	j=[] 
        j=soup.find_all('a')
        for w in j:
            z=w.get('class')
            if(z!=None):
                  if(z[0]=='mw-nextlink' and rr!=1):
                       rr=1 
                       q[i].append('https://en.wikipedia.org'+w.get('href'))
        if(rr==0):
                  q[i].append('0')  

joey=0
while True:
   joey=joey+1
   print(joey) 
   flag=0
   for i in range(len(r)):
        rr=0 
        print(i)
        if(q[i][len(q[i])-1]=='0'):
              continue 
        response=urllib2.urlopen(q[i][len(q[i])-1])
        p=response.read()
        soup=BeautifulSoup(p)
        j=[] 
        j=soup.find_all('a')
        for w in j:
            z=w.get('class')
            if(z!=None):
                  if(z[0]=='mw-nextlink' and rr!=1):
                         rr=1
                         flag=1   
                         q[i].append('https://en.wikipedia.org'+w.get('href'))
        if(rr==0):
                  q[i].append('0')  
   if(flag==0):
        break
                 
nn=open('sitesn.txt','w')
for w in q[i]:
    for ww in w: 
        nn.write(ww.encode('UTF-8')+'\n')      
    nn.write('\n')


