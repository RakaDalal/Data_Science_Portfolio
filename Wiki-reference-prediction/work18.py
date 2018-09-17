import nltk
import urllib2
from bs4 import BeautifulSoup
from socket import error as SocketError


while(True):
      try: 
          response=urllib2.urlopen(url)
          break
      except SocketError as e:
               if e.errno!=errno.ECONNRESET:
                    raise
               pass 
#response=urllib2.urlopen(url)
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

c=0
pp=[]
for v in r:
	url=v
	c=c+1
	#print(c)
	#response=urllib2.urlopen(url)
        while(True):
                try: 
                      response=urllib2.urlopen(url)
                      break
                except SocketError as e:
                      print("ERROR1")
                      pass 
                except urllib2.URLError as err: 
                      print("ERROR2")
                      pass 
	pp.append(response.read())

w18v=0

for i in range(len(r)):
	print(i,len(r))
        refs=0   
	soup=BeautifulSoup(pp[i])
	g=soup.find_all('li')
	for elem in g:
		v=elem.get('id')
		if(v!=None):
		      if(v.startswith('cite')):
			      refs=refs+1
        print("XX   "+str(refs))
        if(refs>=threshold):
                print("C"+str(n))
                a='https://en.wikipedia.org/w/index.php?title='+str(r[i][29:])+'&action=history'
                a2=r[i]
                newr[w20v].append([])
                oldr[w20v].append([])
                newr[w20v][w18v]=[]
                oldr[w20v][w18v]=[] 
                execfile('work19.py')
                w18v=w18v+1

