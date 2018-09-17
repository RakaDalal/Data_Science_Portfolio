import re 
import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
x3=open("/home/raka/stats/exp2012.txt")
x4=open("/home/raka/stats/dat.txt",'w')

l=[]
abst=[]
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
                flag2=0
		ww=w.split('\n#')
                abstract=''
                appen="" 
		for x in ww:
                        #print x 
			if x.startswith('*'):
				appen=x[1:]
                                #print(appen)
                        if x.startswith('@'):
                                flag=1
				h=x[2:]
                                h=' '.join(h.split(','))
                                h=''.join([e for e in h if(e.isalpha() or e==' ')])
                                appen=appen+" "+h
                                #print(appen)
                        if x.startswith('!'):
                                abstract=x[1:]
                        if x.startswith('y'):
                                year=x[1:]
                                if((flag==1 and year.isdigit() and (int)(year)>=2012 and (int)(year)<=2014) or ((not year.isdigit()) and flag==1)):
                                       flag2=1  
                appen=str(re.sub(r'[^\w]',' ',appen))
                appen=appen.lower() 
                if(flag==1 and flag2==1):
                    l.append(appen)
                    abst.append(abstract)
                    index2[i].append(asb.index(w))
                    x4.write('\n'*2+"#ref:-"+appen+'\n'*2)
                    x4.write( "ABSTRACT:-"+abstract)
        else:
           root="disk"+str(i+1)+".gsd0"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
	   for w in asb:
                flag=0
                flag2=0
                appen=""
		ww=w.split('\n#')
                abstract=''
                appen="" 
		for x in ww:
			if x.startswith('*'):
				appen=(x[2:])
                        if x.startswith('@'):
                                flag=1
				h=x[2:]
                                h=' '.join(x.split(','))
                                h=' '.join([e for e in h if(e.isalpha() or e==' ')])
                                appen=appen+" "+h
                        if x.startswith('!'):
                                abstract=x[1:]
                        if x.startswith('y'):
                                year=x[1:]
                                if((flag==1 and year.isdigit() and (int)(year)>=2012 and (int)(year)<=2014) or ((not year.isdigit()) and flag==1)):
                                       flag2=1 
                appen=str(re.sub(r'[^\w]',' ',appen))
                appen=appen.lower()
                if(flag==1 and flag2==1):
                        l.append(appen)
                        abst.append(abstract) 
                        index2[i].append(asb.index(w))
                        x4.write('\n'*2+"#ref:-"+appen+'\n'*2)
                        x4.write( "ABSTRACT:-"+abstract) 


x3=x3.read()
x3=x3.split("GENRE:-")
x3=x3[1:]
k=0
site=[]
text=[]

for x in x3:
    y=x.split("http://")
    for z in y:
        gg=z.split('\n')
        site.append('http://'+str(gg[0]))
        text.append(('\n').join(w for w in gg[3:]))


x2=open("/home/raka/stats/ref2014.txt")
x2=x2.read()
x2=x2.split("GENRE:-")
x2=x2[1:]
g=[]
st=[]
for x in x2:
    y=x.split("***")
    for z in y:
        gg=z.split('\n')[0]
        z=z.split("A^ ")
        z=z[1:]
        st.append(gg)
        g.append(z)

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

    
execfile('work27.py')


    
