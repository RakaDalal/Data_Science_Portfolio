import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
import re
import collections
import nltk
import urllib2
from bs4 import BeautifulSoup

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

ind=[]
xx=[]
yeard=[]
def writee(indi,indapp,pq):
        if(indi<9):
	   root="disk"+str(indi+1)+".gsd00"+str(indi+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#ind')
           asb2=asb[indapp].split('\n')
           print('\n'*2)
           for w in asb2:
               print w
           print('\n'*2)
	   g.write('#ref'+'\n'+pq.encode('UTF-8')+'\n'*2+asb[indapp].encode('UTF-8')+'\n'*3)             
        else:
           root="disk"+str(indi+1)+".gsd0"+str(indi+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#ind')
           asb2=asb[indapp].split('\n')
           print('\n'*2)
           for w in asb2:
               print w
           print('\n'*2)
	   g.write('#ref'+'\n'+pq.encode('UTF-8')+'\n'*2+asb[indapp].encode('UTF-8')+'\n'*3) 

ind=[]
xx=[]
yeard=[]

dev=[]
histavgcit=[]
for i in range(len(h)):
    #g.write('#index'+str(r[i])+'\n'*2)
    print(i,len(h))
    inmap.append([])
    dev.append([])
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
              debug=[str(w),str(i)]
              dev[i].append(debug) 
              writee(indi,indapp,pq)
           else:
              outmap[i].append(w)
    if(len(h[i])>0):
        percentage[w22a.index(w22x)].append(len(inmap[i]))




