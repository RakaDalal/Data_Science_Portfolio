
import re
import collections
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

l=[]
asb=[]
s=[]
ss=[]
index2=[]
for i in range(33):
        index2.append([]) 
        print(i)
        if(i<9):
	   root="disk"+str(i+1)+".gsd00"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
	   for w in asb:
                flag=0
		ww=w.split('\n')
		for x in ww:
                        appen=""
			if x.startswith('#*'):
				appen=x[2:]
                        if x.startswith('#@'):
                                flag=1
				x=x[2:]
                                x=' '.join(x.split(','))
                                x=''.join([e for e in x if(e.isalpha() or e==' ')])
                                appen=appen+" "+x
                appen=str(re.sub(r'[^\w]',' ',appen))
                appen=appen.lower() 
                if(flag==1):
                    l.append(appen)
                    index2[i].append(asb.index(w))
        else:
           root="disk"+str(i+1)+".gsd0"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
	   for w in asb:
                flag=0
                appen=""
		ww=w.split('\n')
		for x in ww:
			if x.startswith('#*'):
				appen=(x[2:])
                        if x.startswith('#@'):
                                flag=1
				x=x[2:]
                                x=' '.join(x.split(','))
                                x=''.join([e for e in x if(e.isalpha() or e==' ')])
                                appen=appen+" "+x
                appen=str(re.sub(r'[^\w]',' ',appen))
                appen=appen.lower()
                if(flag==1):
                        l.append(appen)
                        index2[i].append(asb.index(w))



inmap=[]
outmap=[]
def calc(w):
    maxx=0.0
    ww=w.split(' ')
    for s in l: 
        ss=s.split(' ')
        temp=len(list(set(ww).intersection(set(ss))))
        temp=temp/(len(ss)*1.0)
        if(temp>maxx and len(ss)>=7):
             maxx=temp
             sn=s
             ind1=l.index(s) 
    return (maxx,sn,ind1)
 
def indexx(indl):
    sum=0 
    for i in range(len(index2)):
        sum=sum+len(index2[i])
        if(indl>sum):
           continue
        else:
           indl=indl-sum
           break
    return (i,index2[i][indl-1]) 

 
indl=0
g=open("bettermap.txt",'w')

xx=[]
def writee(indi,indapp,pq):
        if(indi<9):
	   root="disk"+str(indi+1)+".gsd00"+str(indi+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
           asb2=asb[indapp].split('\n')
           print('\n'*2)
           for w in asb2:
               print w
           print('\n'*2)
	   g.write('#ref'+'\n'+pq.encode('UTF-8')+'\n'*2+asb[indapp].encode('UTF-8')+'\n'*3)
           cites=0
           years=0
           lines=asb[indapp].split('\n')
           for line in lines:
               if(line!=None):
                  if(line.startswith('#n')): 
                      line=re.sub("[^0-9]", "",line)
                      if(line!=''):
                          cites=float(line) 
 		  if(line.startswith('#y')):	
                      line=re.sub("[^0-9]", "",line)
                      if(line!=''):
                          years=2013-float(line) 
                      if(cites!=0 and years!=0):   
                          xx.append(cites/years)
                        
        else:
           root="disk"+str(indi+1)+".gsd0"+str(indi+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
           asb2=asb[indapp].split('\n')
           print('\n'*2)
           for w in asb2:
               print w
           print('\n'*2)
	   g.write('#ref'+'\n'+pq.encode('UTF-8')+'\n'*2+asb[indapp].encode('UTF-8')+'\n'*3) 
           cites=0
           years=0
           lines=asb[indapp].split('\n')
           for line in lines:
               if(line!=None):
                  if(line.startswith('#n')): 
                      line=re.sub("[^0-9]", "",line)
                      if(line!=''):
                          cites=float(line) 
 		  if(line.startswith('#y')):	
                      line=re.sub("[^0-9]", "",line)
                      if(line!=''):
                          years=2013-float(line) 
                      if(cites!=0 and years!=0):   
                          xx.append(cites/years)


histavgcit=[]
for i in range(len(h)):
    g.write('#index'+str(r[i])+'\n'*2)
    print(i)
    inmap.append([])
    outmap.append([])
    if(len(h[i])>0):
        for w in h[i]:
           pq=w
           value=0.0
           w=str(re.sub(r'[^\w]',' ',w))
           w=w.lower()
           (value,mapp,indl)=calc(w)
           histavgcit.append(mapp)
           if (value>0.8):
              print('\n'+w+'\n'*2+mapp+'\n')
              (indi,indapp)=indexx(indl+1)
              inmap[i].append(w)
              writee(indi,indapp,pq)
           else:
              outmap[i].append(w)
 
xxx=[]
for w in inmap:
   if(len(w)!=0):
      xxx.append(len(w))


histavgcit=collections.Counter(histavgcit)
x=histavgcit.values()
'''
import matplotlib.pyplot as plt
n,bins,patches = plt.hist(x,20,facecolor='g',alpha=0.75)
plt.ylabel('No of References')
plt.xlabel('No of wikipages')
plt.title('Histogram of references')
plt.axis([0, 30, 0, 500])
plt.grid(True)
plt.show()

plt.figure(1)
n,bins,patches = plt.hist(xx,10,facecolor='g',alpha=0.75)
plt.xlabel('Average citations')
plt.ylabel('No of Papers cited')
plt.title('Average citations')
plt.axis([0, 200, 0, 70])
plt.grid(True)
plt.show()

plt.figure(2)
n,bins,patches = plt.hist(xxx,10,facecolor='g',alpha=0.75)
plt.xlabel('No of wikipages')
plt.ylabel('No of references')
plt.title('Histogram of references')
plt.axis([0, 10, 0, 20])
plt.grid(True)
plt.show()


f.close()
'''             
