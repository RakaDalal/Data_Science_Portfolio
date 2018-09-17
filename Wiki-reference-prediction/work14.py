'''
execfile('work11.py')

execfile('work16.py')

def indexx2(indl):
    sum=0 
    for i in range(len(inmap)):
        sum=sum+len(inmap[i])
        if(indl>sum):
           continue
        else:
           indl=indl-sum
           break
    return (i) 


innmap=[]
for w in inmap:
    for x in w:
       innmap.append(x)
present=[]

for key in citnet.keys():
    present.append(innmap[g.index(key)])


def common(p,q):
     p=str(re.sub(r'[^\w]',' ',p))
     p=p.lower()
     q=str(re.sub(r'[^\w]',' ',q))
     q=q.lower()
     temp=len(list(set(p.split()).intersection(set(q.split())))) 
     return temp/(min(len(p.split()),len(q.split()))*1.0)    

f=open("nohistry.txt",'w')
citedtimes={}
for w in present:
    flag=0
    print(present.index(w))
    citedtimes[citnet.keys()[present.index(w)]]=[]
    val=0.0
    for i in range(len(refname2)):
        for j in range(len(refname2[i])):
		value=0.0
                if(len(refname2[i][j].split())>=8 and len(refname2[i][j].split())<=50):
		    value=common(refname2[i][j],w)
                    if(value>val):
                       val=value
                       stri=refname2[i][j] 
		if (value>0.7):
                    flag=1 
                    print("HOOLLAH")
    		    citedtimes[citnet.keys()[present.index(w)]].append(time2[i][j])
    if(flag==0):
        web=indexx2(innmap.index(w))
        f.write("\n"+str(r[web])+"\n"+str(w)+"\n"*2)  
    print(val)
    print("\n"+stri+"\n"*2+w+"\n")

	 
f=open("tired.txt",'w')
for m in citedtimes.keys():
    for x in citedtimes[m]:
        f.write(m+"   "+x+'\n')   

'''
biggy={}        
for i in range(33):
        print(i)
        if(i<9):
	   root="disk"+str(i+1)+".gsd00"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('\n'*2)
	   for w in asb:
                flag=0
		ww=w.split('\n')
                h=[]
		for x in ww:
                        if x.startswith('#index'):
                                biggy[x[6:]]=[]
                                inde=x[6:] 
			if x.startswith('#y'):
				biggy[inde].append(x[2:])  
        else:
           root="disk"+str(i+1)+".gsd0"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('\n'*2)
	   for w in asb:
                flag=0
		ww=w.split('\n')
                h=[]
		for x in ww:
			if x.startswith('#index'):
                                biggy[x[6:]]=[]
                                inde=x[6:] 
			if x.startswith('#y'):
				biggy[inde].append(x[2:])  
 
