import re
x1=open("/home/raka/stats/information.txt")
x2=open("/home/raka/stats/ref2014.txt")
x3=open("/home/raka/stats/exp2012.txt")
x4=open("/home/raka/stats/ref2012.txt")
x1=x1.read()
x2=x2.read()
x3=x3.read()
x4=x4.read()
import matplotlib.pyplot as plt

def common(p,q):
     p=str(re.sub(r'[^\w]',' ',p))
     p=p.lower()
     q=str(re.sub(r'[^\w]',' ',q))
     q=q.lower()
     temp=len(list(set(p.split()).intersection(set(q.split()))))
     if(min(len(p.split()),len(q.split()))*1.0>6): 
       return temp/(min(len(p.split()),len(q.split()))*1.0)
     else:
       return 0.0


#MAS DATASET
reff=[]
x1=x1.split("GENRE:-")
for x in x1:
    y=x.split("#ref")
    y=y[1:]
    for z in y:
        z=z.split('\nex')
        reff.append(z[0])
        z1=z[1].split('#')
        abstract.append('')
        abstract[len(abstract)-1]=[w[1:] for w in z1 if w.startswith('!')]
        refs.append('')
        refs[len(refs)-1]=[w[1:] for w in z1 if w.startswith('*')]

x2=x2.split("GENRE:-")
x2=x2[1:]
g=[]
for i in range(len(reff)):
    g.append([])
    for x in x2:
        y=x.split("***")
	for z in y:
	    gg=z.split('\n')[0]
	    z=z.split("A^ ")
            z=z[1:]
	    for zz in z:
	        if(common(zz,reff[i])>0.74):
                     print zz
                     print reff[i] 
                     g[i].append(gg)
                     break

x3=x3.split("GENRE:-")
x3=x3[1:]
k=0
for s in g:
    k=k+1 
    text.append([])
    for site in s:
            for x in x3:
                y=x.split("http://")
                for z in y:
                     gg=z.split('\n')
                     if('http://'+str(gg[0])==str(site)):
                           print(k)
                           #print(gg[0])
                           text[len(text)-1].append(('\n').join(w for w in gg[1:]))

for i in range(len(text)):
    text[i]=list(set(text[i])) 


x4=x4.split("GENRE:-")
x4=x4[1:]
k=0
for s in g:
    k=k+1 
    oldrefset.append([])
    for site in s:
            for x in x4:
                y=x.split("http://")
                for z in y:
                     gg=z.split('\n')
                     if('http://'+str(gg[0])==str(site)):
                           print(k)
                           #print(gg[0])
                           oldrefset[len(oldrefset)-1].append(('\n').join(w for w in gg[1:]))

for i in range(len(oldrefset)):
    oldrefset[i]=list(set(oldrefset[i])) 
                 
 
   
 
    

           

        
        	 
