import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
import re
import collections
import nltk
import urllib2
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
g=open("/home/raka/stats/information.txt",'w')
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

w22f=open("/home/raka/stats/ref2014.txt")
percentage=[]
w22f=w22f.read()
w22a=w22f.split("GENRE")
for w22x in w22a:
     g.write('\n'*2+"GENRE:-"+w22x.split('\n')[0]+'\n'*3) 
     h=[]
     percentage.append([])
     w22h=w22x.split("***")
     for w22z in w22h:
         g.write('\n'*2+"WEBSITE:-"+w22z.split('\n')[0]+'\n'*3)
         h.append([])
         w22n=w22z.split("A")
         for w22i in w22n:
             if(w22i.startswith('^')):
                  h[w22h.index(w22z)].append(w22i)
     execfile('work23.py')

for i in range(1,len(percentage)):
        plt.hold(False)  
        z=plt.hist(percentage[i],bins=8,color='green')
	plt.xlabel("percentage of references mapped to MAS")
	plt.ylabel("number of pages of that genre")
	plt.ylim(0,max(z[0])+2)
	plt.savefig("/home/raka/stats/stat6"+str(i)+".png")
